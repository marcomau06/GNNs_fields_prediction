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
import model_buckling1
from utils_buckling2 import*
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.optim.lr_scheduler import ExponentialLR

wDr = os.getcwd()
# Import data from post-buckling
test_num = 10
field_path = wDr + r'\Fields\test' + str(test_num)

train_data, test_data = load_graph(path=field_path, data_types = 6,
                                   steps=101, split =0.9,
                                   shuffle = True)

mean, std = mean_std(train_data)
train_data = transform_stdScaler(mean, std, train_data)
test_data = transform_stdScaler(mean, std, test_data)

# Import data from eigenvalue buckling analysis (FIELD used as node features next)
test_num = 12
field_path = wDr + r'\Fields\test' + str(test_num)

train_data_eigen, test_data_eigen = load_graph(path=field_path, data_types = 6,
                                   steps=101, split =0.9,
                                   shuffle = True)

os.chdir(wDr)

learned_model = model_buckling1.EncodeProcessDecode(
                    node_feat_size = 4,                    
                    edge_feat_size = 6,
                    output_size=2,
                    latent_size=16,                  
                    message_passing_steps=15)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
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

"""Simulation step (-1 == LAST) """
step = -1 
""""""

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
            node_features_tmp1 = train_data_eigen[geom].ground_truth[step,:,:2]
            node_features_tmp2 = train_data[geom].node_features
            
            node_features_tmp = torch.cat([node_features_tmp2, node_features_tmp1], dim = -1)
            
            edge_features1 = train_data[geom].edge_features       
            
            for i in range(0,edge_index.size(1)):            
                xij = node_features_tmp1[edge_index[0][i]-1,:] - node_features_tmp1[edge_index[1][i]-1,:]
                
                mod_xij = torch.linalg.norm(xij).view(1)
                
                tmp = torch.cat((xij,mod_xij)).view(1,3) 
                # tmp = torch.cat((xij/mod_xij,mod_xij)).view(1,3) 
                
                if i==0:
                    edge_features2 = torch.clone(tmp)
                else:                
                    edge_features2 = torch.cat((edge_features2, tmp),dim=0)
                        
            edge_features = torch.cat((edge_features1, edge_features2),dim=1)
            
            fields = train_data[geom].ground_truth           
            
            node_features = normalizer(node_features_tmp)       
            edge_features = normalizer(edge_features)
            
            ground_truth = fields[step,:,:2]                          
                   
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

dataloader = DataLoader(dataset, batch_size = 5, shuffle = True)
# for data in dataloader:
#     print(data.edge_.shape)
            
#%% Training
# import collections
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
epochs = 500

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
def test_f(model, test_data, test_data_len):
   
    model.eval()
        
    loss_all = 0   
    pred = []    
    for geom in range(test_data_len):
        
        # print('Sample:', geom)
        
        edge_index = test_data[geom].edge_index        
        node_features_tmp1 = test_data_eigen[geom].ground_truth[step,:,:2]     
        node_features_tmp2 = test_data[geom].node_features
        
        node_features = torch.cat([node_features_tmp2, node_features_tmp1], dim = -1)
        
        edge_features1 = test_data[geom].edge_features       
        
        for i in range(0,edge_index.size(1)):            
            xij = node_features_tmp1[edge_index[0][i]-1,:] - node_features_tmp1[edge_index[1][i]-1,:]
            
            mod_xij = torch.linalg.norm(xij).view(1)
            
            tmp = torch.cat((xij,mod_xij)).view(1,3) 
            # tmp = torch.cat((xij/mod_xij,mod_xij)).view(1,3) 
            
            if i==0:
                edge_features2 = torch.clone(tmp)
            else:                
                edge_features2 = torch.cat((edge_features2, tmp),dim=0)
                    
        edge_features = torch.cat((edge_features1, edge_features2),dim=1)      
        
        fields = test_data[geom].ground_truth
        elements = test_data[geom].connectivity       
        
        node_features_tmp = normalizer(node_features)       
        edge_features = normalizer(edge_features)                  
               
        data = Data(edge_index = edge_index -1, x=node_features_tmp, edge_attr=edge_features)                 
        graph = data.to(device)
                                
        out = model(graph)
        pred.append(out)
               
        ground_truth = fields[step,:,:2]                                       
        # loss = F.mse_loss(out, ground_truth)
        loss = F.l1_loss(out, ground_truth)
        loss_all += loss.item()                   
        
    return pred, loss_all/test_data_len

# test_data_len = len(test_data)
test_data_len = 10
pred, mae_test = test_f(model, test_data, test_data_len)    
print('MAE on test data',mae_test)
#%% Save model and load model
# Saving
path = wDr + r'\Models'
model_name = r'\buckling_test10_test12_onlyDef.pt'

save_model = False

if save_model == True:
    torch.save(model.state_dict(), path + model_name)

# Loading
load_model = False

if load_model == True:  
    learned_model = model_buckling1.EncodeProcessDecode(
                        node_feat_size = 4,                    
                        edge_feat_size = 6,
                        output_size=2,
                        latent_size=16,                  
                        message_passing_steps=15)
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model = learned_model.to(device)    
    model.load_state_dict(torch.load(path + model_name))
#%% Plotting deformed shapes on test data

meshcolor = 'black'
# Plotting mesh and scalar field
import matplotlib.pyplot as plt

# plot FE mesh
def plot_fem_mesh(nodes_x, nodes_y, elements, meshcolor = 'k'):
    for element in elements:
        xA = nodes_x[element[0]]
        yA = nodes_y[element[0]]
        
        xB = nodes_x[element[1]]
        yB = nodes_y[element[1]]
                      
        plt.plot([xA,xB],[yA,yB],color = meshcolor, linewidth = 2)

def plotting(scale_fac = None, def_shape = True, compon = None, coords = None,
             field = None, elements = None, levels_contour = None, cmap = None, meshcolor = 'k'):
    
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
            
    # fig = plt.figure()
    # plot FE mesh
    plot_fem_mesh(nodes_x, nodes_y, elements, meshcolor = meshcolor)
          
    # # show
    # plt.colorbar()
    plt.axis('equal')
    plt.show()    
    
    # return fig


def plot_test(scale_fac, def_shape, test_data, pred, geom, compon,
              levels_contour, cmap):
    
    elements = test_data[geom].connectivity.detach().numpy()    
    node_features = test_data[geom].node_features        
    node_features = node_features.detach().numpy()
    
    fields = test_data[geom].ground_truth
   
    coords = node_features[:,:2]        
    '''Ground truth'''
    field_truth = fields[step,:,:2]
    # field_truth = inverse_transform_minmax(maxs[2][step,:,:], mins[2][step,:,:], field_truth)
    field_truth = inverse_transform_stdScal(mean[2][step,:,:2], std[2][step,:,:2], field_truth)
    field_truth = field_truth.detach().numpy()     
    # print(field_truth)
    fig = plotting(scale_fac, def_shape, compon, coords,
                    field_truth, elements, levels_contour, cmap, 'r')
       
    '''Prediction'''
    field = pred[geom]        
    # field = inverse_transform_minmax(maxs[2][step,:,:], mins[2][step,:,:], field)
    field = inverse_transform_stdScal(mean[2][step,:,:2], std[2][step,:,:2], field)
    field = field.detach().numpy()    
    # print(field)
    
    fig = plotting(scale_fac, def_shape, compon, coords,
                   field, elements, levels_contour, cmap, 'k')
       
# Choose geometry
geom = 9

# Levels in the contour field to plot
# levels_contour = np.linspace(-1.6,1.6,100)
levels_contour = 100

# Color map fields
cmap = 'viridis'

# Choose field to plot
compon = 0

def_shape = True
scale_fac = 1.0

fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')

plot_test(scale_fac, def_shape, test_data, pred, geom, compon,
          levels_contour, cmap)

#%% Error on the fields prediction (metrics)
# plot FE mesh
def plot_fem_mesh(nodes_x, nodes_y, elements, field):
    
    stress = field
    s11_el = []
    for element in elements:
        s11_el.append((stress[element[0]] + stress[element[1]])/2)   
    
    norm = matplotlib.colors.Normalize(vmin=min(s11_el), vmax=max(s11_el), clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='viridis')    
    
    fig, ax = plt.subplots(figsize=(4, 4))
    for count,element in enumerate(elements):
        xA = nodes_x[element[0]]
        yA = nodes_y[element[0]]
        
        xB = nodes_x[element[1]]
        yB = nodes_y[element[1]]
                      
        plt.plot([xA,xB],[yA,yB],color = mapper.to_rgba(s11_el[count]), linewidth = 2)        
    
    fig.colorbar(mapper, orientation='vertical', label=r'$u_{1}$')
    
    
def plotting(scale_fac = None, def_shape = True, compon = None, coords = None,
             field = None, elements = None, levels_contour = None, cmap = None, meshcolor = 'k'):
    
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
    
    # plot FE mesh
    plot_fem_mesh(nodes_x, nodes_y, elements, field)

def critical_reg_plot(pl_fig, scale_fac, def_shape, geom, compon, test_data, pred, rel_error_thrd = None, levels_contour = None):
    
    elements = test_data[geom].connectivity.detach().numpy()
    node_features = test_data[geom].node_features             
    node_features = node_features.detach().numpy()
    coords = node_features[:,:2]  
    
    '''Ground truth'''
    fields = test_data[geom].ground_truth
    field_truth = fields[step,:,compon]
    # field_truth = inverse_transform_minmax(maxs[2][step,:,:], mins[2][step,:,:], field_truth)
    field_truth = inverse_transform_stdScal(mean[2][step,:,compon], std[2][step,:,compon], field_truth)
    field_truth = field_truth.detach().numpy()    
    
    '''Predictions'''
    field = pred[geom][:,compon]        
    # field = inverse_transform_minmax(maxs[2][step,:,:], mins[2][step,:,:], field)
    field = inverse_transform_stdScal(mean[2][step,:,compon], std[2][step,:,compon], field)
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
    truth_field = test_data[geom].ground_truth[step,:,compon]
    # truth_field = inverse_transform_minmax(maxs[2][step,:,compon], mins[2][step,:,compon], truth_field)
    truth_field = inverse_transform_stdScal(mean[2][step,:,compon], std[2][step,:,compon], truth_field)
    truth_field = truth_field.detach().numpy() 
    
    '''Predictions'''    
    pred_field = pred[geom]
    # pred_field = inverse_transform_minmax(maxs[2][step,:,:], mins[2][step,:,:], pred_field)    
    pred_field = inverse_transform_stdScal(mean[2][step,:,:2], std[2][step,:,:2], pred_field)    
    pred_field = pred_field[:,compon].detach().numpy()       
    
    return coords, elements, (pred_field - truth_field)
    
# Deformed shape OFF
def_shape = False
scale_fac = 1.5

# Geometry
geom = 0

# Choose field output
compon = 1

# Field error map
cmap = 'coolwarm'
levels_contour = 100

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
    
    







    
    