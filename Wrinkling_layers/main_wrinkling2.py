# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 17:09:31 2022

@author: marcomau
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import model_wrinkling2 as model
from utils_wrinkling2 import load_graph, mean_std, transform_stdScaler, inverse_transform_stdScal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.optim.lr_scheduler import ExponentialLR

wDr = os.getcwd()
#%% Import data
test_num = 9
field_path = wDr + r'\Fields\test' + str(test_num)

# Base material properties
Ei_arr = np.linspace(10000,200000,num=100)
nui_arr = np.linspace(0.01,0.49,num=5)
comb_params = np.array(np.meshgrid(Ei_arr, nui_arr)).T.reshape(-1,2)

data = load_graph(path=field_path, data_types = 6,
                                   steps=1, split =None,
                                   shuffle = True, comb_params = comb_params)
os.chdir(wDr)
#%% Data pre-processing
train_data, test_data = train_test_split(data, train_size=0.9, random_state=42, shuffle=True)    

mean, std = mean_std(train_data)
train_data = transform_stdScaler(mean, std, train_data)
test_data = transform_stdScaler(mean, std, test_data)

# Delete strain component LE33 (=0) for plain strain
mean1 = mean[2][:,:4]
mean2 = mean[2][:,5:]
mean = torch.cat((mean1,mean2),dim = -1)  

std1 = std[2][:,:4]
std2 = std[2][:,5:]
std = torch.cat((std1,std2),dim = -1)  
#%% Model
learned_model = model.EncodeProcessDecode(
                    node_feat_size = 4,                    
                    edge_feat_size = 3,
                    output_size=5,
                    latent_size=32,                  
                    message_passing_steps=15)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
model = learned_model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
scheduler = ExponentialLR(optimizer, gamma=0.9)

print('Learnable parameters',sum(p.numel() for p in model.parameters()))
#%% Normalizer
def normalizer(features):
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    features = scaler.fit_transform(features.detach().numpy())
    features = torch.tensor(features).type(torch.FloatTensor) 
    return features
#%% Data-set creation

from torch_geometric.data import InMemoryDataset

class data_set(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def processed_file_names(self):
        return ['data_set.pt']

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        for geom in range(len(train_data)):                                 
            
            edge_index = train_data[geom].edge_index
            node_features_tmp = train_data[geom].node_features            
            edge_features = train_data[geom].edge_features       
            fields = train_data[geom].ground_truth                                  
                        
            node_features = normalizer(node_features_tmp)       
            edge_features = normalizer(edge_features)                
            
            # Delete strain component LE33 (=0) for plain strain
            gt1 = fields[:,:4]
            gt2 = fields[:,5:]
            fields = torch.cat((gt1,gt2),dim = -1)                    
            ground_truth = fields                                        
                   
            data = Data(edge_index = edge_index -1, x=node_features, edge_attr=edge_features, y = ground_truth)         
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])    
dataset = data_set(root = wDr + r'\Datasets')
#%% Mini-batching
from torch_geometric.loader import DataLoader

dataloader = DataLoader(dataset, batch_size = 1, shuffle = True)           
#%% Training
import time

def train(model, dataloader):
    model.train()        
    
    start_time = time.time()
    loss_all = 0    
    for data in dataloader:                                          
        
        graph = data.to(device)      
        out = model(graph).to(device) 
           
        ground_truth = graph.y
        # loss = F.mse_loss(out, ground_truth).to(device)
        loss = F.l1_loss(out, ground_truth).to(device) 
        loss.backward()
        optimizer.step()       
        
        optimizer.zero_grad()                                       
        
        loss_all += loss.item()       
        # print('Loss sample MAE:', loss.item())       
        
    # print('Time per sample', (time.time() - start_time))
    return loss_all/len(dataloader)

loss_hist = []
epochs = 100

for epoch in range(epochs):    
    start_time = time.time()       
    print('Epoch:', epoch)
    
    loss_avg = train(model, dataloader)    
    print('Loss epoch:', loss_avg)    
    
    loss_hist.append(loss_avg)       
    print('Time per epoch', (time.time() - start_time))
    
    scheduler.step()
#%% Training history
import matplotlib.pyplot as plt

plt.figure()

plt.plot(loss_hist)
plt.xlabel('Epoch')
plt.ylabel('MAE')
#%% Testing
def test_f(model, test_data, geom):   
    model.eval()
    
    pred = []
    loss_all = 0       
   
    edge_index = test_data[geom].edge_index
    node_features = test_data[geom].node_features            
    edge_features = test_data[geom].edge_features        
    fields = test_data[geom].ground_truth            
    
    
    node_features = normalizer(node_features)
    edge_features = normalizer(edge_features)                  
           
    data = Data(edge_index = edge_index -1, x=node_features, edge_attr=edge_features)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph = data.to(device)
                            
    start_time = time.time() 
    out = model(graph)
    print('Inference', (time.time() - start_time))
    
    device = torch.device('cpu')
    
    # Delete strain component LE33 (=0) for plain strain
    gt1 = fields[:,:4]
    gt2 = fields[:,5:]
    fields = torch.cat((gt1,gt2),dim = -1)      
    ground_truth = fields
    
    # loss = F.mse_loss(out, ground_truth)
    loss = F.l1_loss(out.to(device), ground_truth)
    loss_tmp += loss.item()                                               
   
    loss_all += loss_tmp       
    pred.append(out.to(device))              
        
    # return pred, loss_all/test_data_len
    return pred, loss_all

# test_data_len = len(test_data)
geom = 1
pred, mae_test = test_f(model, test_data, geom)    
print('MAE on test data',mae_test)
#%% Save model and load model
# Saving
path = wDr + r'\Models'
model_name = r'\wrinkl_latent32_L15.pt'

save_model = False
# save_model = True

if save_model == True:
    torch.save(model.state_dict(), path + model_name)

# Loading
load_model = False

if load_model == True:  
    learned_model = model.EncodeProcessDecode(
                        node_feat_size = 4,                    
                        edge_feat_size = 3,
                        output_size=5,
                        latent_size=32,                  
                        message_passing_steps=15)
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model = learned_model.to(device)    
    model.load_state_dict(torch.load(path + model_name))
#%% Delete post-processed training data from the folder
import shutil
path_post = wDr + r'\Datasets\processed'
if os.path.isdir(path_post):
    shutil.rmtree(path_post)         
#%% Plotting on test data
meshcolor = 'black'
# Plotting mesh and scalar field
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# converts quad elements into tri elements for plotting
def quads_to_tris(quads):
    tris = [[None for j in range(3)] for i in range(2*len(quads))]
    for i in range(len(quads)):
        j = 2*i
        n0 = quads[i][0]
        n1 = quads[i][1]
        n2 = quads[i][2]
        n3 = quads[i][3]
        tris[j][0] = n0
        tris[j][1] = n1
        tris[j][2] = n2
        tris[j + 1][0] = n2
        tris[j + 1][1] = n3
        tris[j + 1][2] = n0
    return tris

# plots finite element mesh
def plot_fem_mesh(nodes_x, nodes_y, elements, meshcolor = 'black'):
    
    fig, ax = plt.subplots(figsize=(4, 4))  
    fig.patch.set_visible(False)
    ax.axis('off')
    for element in elements:
        x = [nodes_x[element[i]] for i in range(len(element))]
        y = [nodes_y[element[i]] for i in range(len(element))]
        ax.fill(x, y, edgecolor=meshcolor, fill=False)

def plotting(scale_fac = None, def_shape = True, compon = None, coords = None, field = None, elements = None, levels_contour = None, cmap = None):
    
    if def_shape == True:
        nodes_x = coords[:,0] + scale_fac * field[:,0]
        nodes_y = coords[:,1] + scale_fac * field[:,1]
    else:        
        nodes_x = coords[:,0]
        nodes_y = coords[:,1]
    elements = elements - 1
    
    if compon != None:
        nodal_values = field[:,compon]
    else:
        nodal_values = field    
    
    # convert all elements into triangles
    elements_all_tris = quads_to_tris(elements)
    
    # create an unstructured triangular grid instance
    triangulation = tri.Triangulation(nodes_x, nodes_y, elements_all_tris)
    
    # fig = plt.figure()
    # plot FE mesh
    plot_fem_mesh(nodes_x, nodes_y, elements, meshcolor = meshcolor)
    
    # plot the contours
    plt.tricontourf(triangulation, nodal_values,levels = levels_contour, cmap=cmap)
    
    # show
    plt.colorbar()
    plt.axis('equal')
    plt.show()    
    
    return fig


def plot_test(scale_fac, def_shape, test_data, pred, geom, compon,
              levels_contour, cmap):
    
    elements = test_data[geom].connectivity.detach().numpy()    
    node_features = test_data[geom].node_features        
    node_features = node_features.detach().numpy()
    
    fields = test_data[geom].ground_truth
   
    # Delete strain component LE33 (=0) for plain strain
    gt1 = fields[:,:4]
    gt2 = fields[:,5:]
    fields = torch.cat((gt1,gt2),dim = -1)     
   
    coords = node_features[:,:2]        
    '''Ground truth'''
    field_truth = fields
    # field_truth = inverse_transform_minmax(maxs[2][step,:,:], mins[2][step,:,:], field_truth)
    field_truth = inverse_transform_stdScal(mean, std, field_truth)
    field_truth = field_truth.detach().numpy()     
    
    fig = plotting(scale_fac, def_shape, compon, coords,
                   field_truth, elements, levels_contour, cmap)
       
    '''Prediction'''
    field = pred[geom]        
    # field = inverse_transform_minmax(maxs[2][step,:,:], mins[2][step,:,:], field)
    field = inverse_transform_stdScal(mean, std, field)
    field = field.detach().numpy()    
    
    fig = plotting(scale_fac, def_shape, compon, coords,
                   field, elements, levels_contour, cmap)
               
# Choose geometry
geom = 0

# Levels in the contour field to plot
# levels_contour = np.linspace(-0.25,0.25,100)
levels_contour = 100

# Color map fields
cmap = 'viridis'

# Choose field to plot
compon = 2

def_shape = True
scale_fac = 1.0
plot_test(scale_fac, def_shape, test_data, pred, geom, compon,
          levels_contour, cmap)
#%% Error on the fields prediction (metrics)

def critical_reg_plot(pl_fig, scale_fac, def_shape, geom, compon, test_data, pred, rel_error_thrd = None, levels_contour = None):
    
    elements = test_data[geom].connectivity.detach().numpy()
    node_features = test_data[geom].node_features             
    node_features = node_features.detach().numpy()
    coords = node_features[:,:2]  
    
    '''Ground truth'''
    fields = test_data[geom].ground_truth
    # Delete strain component LE33 (=0) for plain strain
    gt1 = fields[:,:4]
    gt2 = fields[:,5:]
    fields = torch.cat((gt1,gt2),dim = -1)         
    
    field_truth = fields[:,compon]
    # field_truth = inverse_transform_minmax(maxs[2][step,:,:], mins[2][step,:,:], field_truth)
    field_truth = inverse_transform_stdScal(mean[:,compon], std[:,compon], field_truth)
    field_truth = field_truth.detach().numpy()    
    
    '''Predictions'''
    field = pred[geom][:,compon]        
    # field = inverse_transform_minmax(maxs[2][step,:,:], mins[2][step,:,:], field)
    field = inverse_transform_stdScal(mean[:,compon], std[:,compon], field)
    field = field.detach().numpy() 
    
    error_map = np.abs((field - field_truth)/field_truth)    
    
    accurate_regions=np.where(error_map <= rel_error_thrd)[0]
    critical_regions=np.where(error_map > rel_error_thrd)[0]
    
    # error_map[accurate_regions] = 0
    # error_map[critical_regions] = 1    
    
    if pl_fig == True:
        fig = plotting(scale_fac, def_shape, None, coords, error_map, elements, levels_contour, cmap)

    return len(critical_regions)/len(error_map)

def error_map_f(geom, compon, test_data, pred):
    
    elements = test_data[geom].connectivity.detach().numpy()
    node_features = test_data[geom].node_features              
    node_features = node_features.detach().numpy()
    coords = node_features[:,:2]  
    
    '''Ground truth'''
    fields = test_data[geom].ground_truth
    # Delete strain component LE33 (=0) for plain strain
    gt1 = fields[:,:4]
    gt2 = fields[:,5:]
    fields = torch.cat((gt1,gt2),dim = -1)         
    
    truth_field = fields[:,compon]
    
    # truth_field = inverse_transform_minmax(maxs[2][step,:,compon], mins[2][step,:,compon], truth_field)
    truth_field = inverse_transform_stdScal(mean[:,compon], std[:,compon], truth_field)
    truth_field = truth_field.detach().numpy() 
    
    '''Predictions'''    
    pred_field = pred[geom]
    # pred_field = inverse_transform_minmax(maxs[2][step,:,:], mins[2][step,:,:], pred_field)    
    pred_field = inverse_transform_stdScal(mean, std, pred_field)    
    pred_field = pred_field[:,compon].detach().numpy()       
    
    return coords, elements, (pred_field - truth_field)
    
# Deformed shape OFF
def_shape = False
scale_fac = 1.0

# Geometry
geom = 1

# Choose field output
compon = 4

# Field error map
cmap = 'coolwarm'
levels_contour = np.linspace(-0.42,0.42,100)
# levels_contour = 100

coords, elements, error_map = error_map_f(geom, compon, test_data, pred)
fig = plotting(scale_fac, def_shape, None, coords, error_map, elements, levels_contour, cmap)

# Relative error field map with threshold
cmap = 'Greys'
pl_fig = True
critical_nodes_fraction = critical_reg_plot(pl_fig, scale_fac, def_shape, geom,
                                            compon, test_data,
                                            pred, rel_error_thrd = 0.04,
                                            levels_contour = 100)
print('Critical node fraction', critical_nodes_fraction*100, '%')
#%% Deformed shapes
fig, ax = plt.subplots(figsize=(4, 4))  

def plot_fem_mesh(nodes_x, nodes_y, elements, meshcolor = 'black'):
        
    fig.patch.set_visible(False)
    ax.axis('off')
    for element in elements:
        x = [nodes_x[element[i]] for i in range(len(element))]
        y = [nodes_y[element[i]] for i in range(len(element))]
        ax.fill(x, y, edgecolor=meshcolor, fill=False)

def def_shapes_plot(scale_fac, geom, test_data, disp_field, def_shape = True):   
    
    elements = test_data[geom].connectivity.detach().numpy()
    node_features = test_data[geom].node_features.detach().numpy()  
    coords = node_features[:,:2]      
    
    if def_shape == True:
        coords_new = coords + scale_fac * disp_field
    else:
        coords_new = coords
    
    nodes_x = coords_new[:,0]
    nodes_y = coords_new[:,1]       
    elements = elements - 1    
    
    # plt.figure()
    # plot deformed shape
    plot_fem_mesh(nodes_x, nodes_y, elements, meshcolor = meshcolor)
    plt.axis('equal')
    plt.show()    

# Scale factor plotting deformed shape
scale_fac = 1.0

# Geometry
geom = 0

'''Ground truth'''
truth_field = test_data[geom].ground_truth[:,:2]
# truth_field = inverse_transform_minmax(maxs[2][step,:,compon], mins[2][step,:,compon], truth_field)
truth_field = inverse_transform_stdScal(mean[:,:2], std[:,:2], truth_field)
truth_field = truth_field.detach().numpy() 

        
'''Predictions'''    
pred_field = pred[geom]
pred_field = inverse_transform_stdScal(mean, std, pred_field)    
pred_field = pred_field[:,:2].detach().numpy()     

meshcolor = 'red'
# meshcolor = 'green'
def_shapes_plot(scale_fac, geom, test_data, truth_field, def_shape = True)

meshcolor = 'black'
def_shapes_plot(scale_fac, geom, test_data, pred_field, def_shape = True)

#%% Average relative error on the test dataset
def_shape = False
pl_fig = False
compon = 5
cr_node_fraction = 0
for geom in range(len(test_data)):   
    
    cr_nd_fr = critical_reg_plot(pl_fig, scale_fac, def_shape, geom,
                                            compon, test_data,
                                            pred, rel_error_thrd = 0.04,
                                            levels_contour = 2)
    cr_node_fraction += cr_nd_fr
cr_node_fraction /= len(test_data)
print('Average critical node fraction', cr_node_fraction*100)    
    
    







    
    