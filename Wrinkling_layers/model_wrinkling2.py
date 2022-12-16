# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 11:55:42 2022

@author: marcomau
"""
# GNN for displacement/stress field predictions

import torch
import torch_geometric
from torch.nn import Sequential, Linear, ReLU, LayerNorm
from torch_geometric.nn import MessagePassing
import functools
import collections
from torch_geometric.data import Data

class GraphNetBlock(MessagePassing):
    """Message passing."""
    
    def __init__(self,latent_size, in_size1, in_size2): 
        super(GraphNetBlock, self).__init__(aggr='add')        
        self._latent_size = latent_size
        
        # First net (MLP): eij' = f1(xi, xj, eij)
        self.edge_net = Sequential(Linear(in_size1,self._latent_size),
                                   ReLU(),
                                   Linear(self._latent_size,self._latent_size),
                                   ReLU(),
                                   LayerNorm(self._latent_size))        
        
        # Second net (MLP): xi' = f2(xi, sum(eij'))
        self.node_net = Sequential(Linear(in_size2,self._latent_size),
                                   ReLU(),
                                   Linear(self._latent_size,self._latent_size),
                                   ReLU(),
                                   LayerNorm(self._latent_size))       
        
    def forward(self, graph):
        
        edge_index = graph.edge_index        
        x = graph.x
        edge_features = graph.edge_attr
        
        # Node update
        new_node_features = self.propagate(edge_index, x= x, edge_attr = edge_features)        
        
        # Edge update
        row, col = edge_index
        new_edge_features = self.edge_net(torch.cat([x[row], x[col], edge_features], dim=-1))
        
        # Add residuals
        new_node_features = new_node_features + graph.x
        new_edge_features = new_edge_features + graph.edge_attr       
                
        return Data(edge_index = edge_index, x = new_node_features, edge_attr = new_edge_features)        
    
    def message(self, x_i, x_j, edge_attr):            
        features = torch.cat([x_i, x_j, edge_attr], dim=-1)        
        
        return self.edge_net(features)
    
    def update(self, aggr_out, x):
        # aggr_out has shape [num_nodes, out_channels]        
        tmp = torch.cat([aggr_out, x], dim=-1)                
       
        # Step 5: Return new node embeddings.        
        return self.node_net(tmp)
    
class EncodeProcessDecode(torch.nn.Module):
    """Encode-Process-Decode GraphNet model."""

    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 output_size,
                 latent_size,                 
                 message_passing_steps,
                 name='EncodeProcessDecode'):
        super(EncodeProcessDecode, self).__init__()      
        self._node_feat_size = node_feat_size
        self._edge_feat_size = edge_feat_size
        self._latent_size = latent_size
        self._output_size = output_size      
        self._message_passing_steps = message_passing_steps       
        
        # Encoding net (MLP) for node_features
        self.node_encode_net = Sequential(Linear(self._node_feat_size,self._latent_size),
                         ReLU(),
                         Linear(self._latent_size,self._latent_size),
                         ReLU(),
                         LayerNorm(self._latent_size))               
               
        # Encoding net (MLP) for edge_features
        self.edge_encode_net = Sequential(Linear(self._edge_feat_size,self._latent_size),
                         ReLU(),
                         Linear(self._latent_size,self._latent_size),
                         ReLU(),
                         LayerNorm(self._latent_size))              
        
        # Decoding net (MLP) for node_features (output)
        # ND: "Node features Decoding"
        self.node_decode_net = Sequential(Linear(self._latent_size,self._latent_size),
                        ReLU(),
                        Linear(self._latent_size,self._output_size))
                        
       
        # Processor
        self.message_pass = GraphNetBlock(self._latent_size, self._latent_size*3, self._latent_size*2)
    
    def forward(self, graph):
        """Encodes and processes a graph, and returns node features."""  
        edge_index = graph.edge_index                       
        # x = graph.node_features
        # edge = graph.edge_features
        x = graph.x
        edge = graph.edge_attr
        
        # Encoding node features
        node_latents = self.node_encode_net(x)          
        
        # Encoding edge features
        edge_latents = self.edge_encode_net(edge)        
       
        # latent_graph = Graph(edge_index, node_latents, edge_latents)
        latent_graph = Data(edge_index = edge_index, x = node_latents, edge_attr = edge_latents)                
        
        for _ in range(self._message_passing_steps):
             # latent_graph = GraphNetBlock(self._latent_size, self._latent_size*3, self._latent_size*2)(latent_graph)
             latent_graph = self.message_pass(latent_graph)
        
        """Decodes node features from graph."""   
        # Decoding node features
        decoded_nodes = self.node_decode_net(latent_graph.x)    
        
        return decoded_nodes
  


