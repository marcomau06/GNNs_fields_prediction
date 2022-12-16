# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 11:00:55 2022

@author: marcomau
"""
import os
import time
import copy
import datetime
import json
import torch
import collections
from sklearn.model_selection import train_test_split
import numpy as np

def var_data(features, mean_val):  
    if len(features.size()) == 2:
        dim = 0
    else:
        dim = 1            
    
    return torch.mean(torch.square((features - mean_val)), dim =dim, keepdim = True)
    
def mean_std(data):    
    node_mean = 0
    edge_mean = 0
    gt_mean = 0    
    for geom in range(len(data)):
        
        nd_f = data[geom].node_features
        edg_ft = data[geom].edge_features
        gt = data[geom].ground_truth
        
        node_mean = node_mean + torch.mean(nd_f, dim = 0, keepdim = True)
        edge_mean = edge_mean + torch.mean(edg_ft, dim = 0, keepdim = True)
        gt_mean = gt_mean + torch.mean(gt, dim = 1, keepdim=True)
    node_mean = node_mean/len(data)
    edge_mean = edge_mean/len(data)
    gt_mean = gt_mean/len(data)
    
    ss_node = 0
    ss_edge = 0
    ss_gt = 0
    for geom in range(len(data)):
        nd_f = data[geom].node_features
        edg_ft = data[geom].edge_features
        gt = data[geom].ground_truth
        
        ss_node = ss_node + var_data(nd_f, node_mean)
        ss_edge = ss_edge + var_data(edg_ft, edge_mean)
        ss_gt = ss_gt + var_data(gt, gt_mean)
    ss_node = torch.sqrt(ss_node/len(data))
    ss_edge = torch.sqrt(ss_edge/len(data))
    ss_gt = torch.sqrt(ss_gt/len(data))
    
    means = [node_mean, edge_mean, gt_mean]
    stds = [ss_node, ss_edge, ss_gt]    
    return means, stds
    
def transform_stdScaler(means, stds, data):   
    
    for geom in range(len(data)):       
        gt = data[geom].ground_truth       
        
        gt = (gt - means[2]) / stds[2]        
        data[geom] = data[geom]._replace(ground_truth = gt)
    
    return data

def inverse_transform_stdScal(mean, std, features):       
    return features * std + mean

def min_max(data):
    
    for geom in range(len(data)):
        node = data[geom].node_features
        edge = data[geom].edge_features
        gt = data[geom].ground_truth
        
        node_max = torch.max(node, dim = 0, keepdim = True)[0]
        node_min = torch.min(node, dim = 0, keepdim = True)[0]       
    
        edge_max = torch.max(edge, dim = 0, keepdim = True)[0]
        edge_min = torch.min(edge, dim = 0, keepdim = True)[0]
    
        gt_max = torch.max(gt, dim = 1, keepdim = True)[0]
        gt_min = torch.min(gt, dim = 1, keepdim = True)[0]
        
        if geom == 0:
            node_max_all = node_max
            node_min_all = node_min
            
            edge_max_all = edge_max
            edge_min_all = edge_min
            
            gt_max_all = gt_max
            gt_min_all = gt_min
            
        else:            
            node_max_all = torch.cat([node_max_all, node_max], dim = 0)
            node_min_all = torch.cat([node_min_all, node_min], dim = 0)
        
            edge_max_all = torch.cat([edge_max_all, edge_max], dim = 0)
            edge_min_all = torch.cat([edge_min_all, edge_min], dim = 0)            
            
            gt_max_all = torch.cat([gt_max_all, gt_max], dim = 1)
            gt_min_all = torch.cat([gt_min_all, gt_min], dim = 1)
    
        
    node_max = torch.max(node_max_all, dim = 0, keepdim = True)[0]
    node_min = torch.min(node_min_all, dim = 0, keepdim = True)[0]
    
    edge_max = torch.max(edge_max_all, dim = 0, keepdim = True)[0]
    edge_min = torch.min(edge_min_all, dim = 0, keepdim = True)[0]

    gt_max = torch.max(gt_max_all, dim = 1, keepdim = True)[0]
    gt_min = torch.min(gt_min_all, dim = 1, keepdim = True)[0]
    
    # return [node_max, node_min], [edge_max, edge_min], [gt_max, gt_min]
    return [node_max, edge_max, gt_max], [node_min, edge_min, gt_min]

def min_max_scaler(maxs, mins, data):
    
    for geom in range(len(data)):       
        gt = data[geom].ground_truth       
        
        gt = (gt - mins[2]) / (maxs[2] - mins[2])        
        data[geom] = data[geom]._replace(ground_truth = gt)
    
    return data

def inverse_transform_minmax(_max, _min, features):       
    return features * (_max - _min) + _min

def load_graph(path=None, data_types=None, steps=101, split =None ,shuffle = None):
    os.chdir(path)
    
    files = os.listdir(path)
    num_files = int(len(files)/data_types)  # data_types describes number of data types in the folder  
    
    
    data = []
    graph = collections.namedtuple('graph', ['edge_index', 'node_features',
                                             'edge_features',
                                             'connectivity',
                                             'ground_truth'])
    
    for geomind in range(num_files):
        print(geomind)
        with open('coords'+str(geomind)+'.txt') as json_file:
            coords = json.load(json_file)
        # coords = np.array(coords)        
        coords = torch.tensor(coords)
        coords = coords[:,:2]   
                
        # with open('node_type'+str(geomind)+'.txt') as json_file:
        #     node_type = json.load(json_file)
        # node_type = torch.tensor(node_type)
        
        with open('steps'+str(geomind)+'.txt') as json_file:
            steps_sim = json.load(json_file)
        steps_sim = torch.tensor(steps_sim)        
        
        with open('edge_index'+str(geomind)+'.txt') as json_file:
            edge_index = json.load(json_file)
        # edge_index = np.array(edge_index)
        edge_index = torch.tensor(edge_index)
        
        with open('connectivity'+str(geomind)+'.txt') as json_file:
            elements = json.load(json_file)
        # elements = np.array(elements)
        elements = torch.tensor(elements)
                
        with open('field1-'+str(geomind)+'.txt') as json_file:
            field1 = json.load(json_file)
        # field = np.array(field)
        field1 = torch.tensor(field1)                
        
        with open('field2-'+str(geomind)+'.txt') as json_file:
            field2 = json.load(json_file)
        # field = np.array(field)
        field2 = torch.tensor(field2)    
        
        if steps_sim.item() < steps:
            continue        
        
        # Add "virtual" connections for each node with nodes within a sphere of radius r_v
        # rv = 1.0      
        # visited = []
        # edge_index_virt = torch.zeros((1,2))
        # for i in range(coords.size(0)):
        #     for j in range(coords.size(0)):
        #         if i != j and (((i,j) or (j,i)) not in visited):                
        #             visited.append((i,j))
        #             visited.append((j,i))
        #             dist = (coords[i] - coords[j]).pow(2).sum().sqrt()        
        #             if dist.item() < rv:
        #                 ind1 = np.array([(torch.tensor([i+1, j+1])==k).all() for k in edge_index])                        
        #                 if not(ind1.any()):                            
        #                     edge_index_virt = torch.cat((edge_index_virt, torch.tensor([[i+1, j+1]]),torch.tensor([[j+1, i+1]])),dim=0)
        
        # edge_index_virt = edge_index_virt[1:,:]
        # edge_index = torch.cat((edge_index, edge_index_virt),dim=0)
        
        # # Add "virtual" connections randomly        
        # for i in range(coords.size(0)):
        #     all_node = list(range(coords.size(0)))
        #     all_node.remove(i)            
        #     j = np.random.choice(all_node,size = 2, replace = False)            
        #     edge_index = torch.cat((edge_index, torch.tensor([[i+1, j[0]+1]]), torch.tensor([[j[0]+1, i+1]])),dim=0)    
        #     edge_index = torch.cat((edge_index, torch.tensor([[i+1, j[1]+1]]), torch.tensor([[j[1]+1, i+1]])),dim=0)    
        
        # Compute edge_features from nodal coordinates        
        for i in range(0,edge_index.size(0)):            
            xij = coords[edge_index[i][0]-1,:] - coords[edge_index[i][1]-1,:]
            
            mod_xij = torch.linalg.norm(xij).view(1)
            
            tmp = torch.cat((xij,mod_xij)).view(1,3) 
            # tmp = torch.cat((xij/mod_xij,mod_xij)).view(1,3) 
            
            if i==0:
                edge_features = torch.clone(tmp)
            else:                
                edge_features = torch.cat((edge_features, tmp),dim=0)
         
        
         
        # Concatenate node features without BCs
        # node_features = torch.cat([coords, node_type], dim=-1)             
        node_features = coords
        
        # Concatenate fields as ground truth        
        field = torch.cat([field1, field2[:,:,:1]], dim=-1)      
        
                    
        # Create a collection with graph elements
        tmp = graph(edge_index.t(), node_features, edge_features,
                    elements, field)        
                
        data.append(tmp)     
    
    train_data, test_data = train_test_split(data, train_size=split, random_state=42, shuffle=shuffle)    
    
    return train_data, test_data
    
    
 
    
    
    
    
