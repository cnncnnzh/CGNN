# -*- coding: utf-8 -*-
"""
Created on Wed May 31 23:34:03 2023
"""

import torch
from torch.nn import Linear, ModuleList, ReLU, BatchNorm1d, Dropout
from torch_geometric.utils import softmax
from torch_geometric.nn import global_mean_pool, Set2Set, GCNConv, DiffGroupNorm
from torch_geometric.utils import add_self_loops, degree

    
class MLP(torch.nn.Module):
    '''
    multilayer perceptron after message passing layers
    '''
    def __init__(self,
                 in_dim,
                 n_linear):
        super(MLP, self).__init__()
        self.lin_list = ModuleList()
        for _ in range(n_linear):
            self.lin_list.append(Linear(in_dim, in_dim))
        self.act = ReLU()
        self.out_1 = Linear(in_dim, 32)
        self.out_2 = Linear(32, 1)
        
    def forward(self, x):
        for lin in self.lin_list:
            x = lin(x)
            x = self.act(x)
        x = self.out_1(x)
        x = self.act(x)
        x = self.out_2(x)
        # x = self.act(x)
        return x.reshape(1,-1)


class Global_Attention(torch.nn.Module):
    def __init__(self, dim, n_layers):
        super(Global_Attention, self).__init__()
        
        self.in_layer = torch.nn.Linear(dim + 107, dim)
        
        self.lin_list = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.lin_list.append(torch.nn.Linear(dim, dim))
        
        self.out_layer = torch.nn.Linear(dim, 1)
        self.act = ReLU()
        
    def forward(self, x, global_info, batch):
        x = torch.cat((x, global_info), dim=-1)
        x = self.in_layer(x)
        x = self.act(x)
        for lin in self.lin_list:
            x = lin(x)
            x = self.act(x)
        x = self.out_layer(x)
        return softmax(x, batch)
            
class SKIP(torch.nn.Module):
    '''
    main graph concolutional neuron network class
    '''
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim=64,
        n_conv_layer=5,
        n_linear=1,
        dropout_rate=0.2,
    ):
        super(SKIP, self).__init__()

        # setup linear layer before gnn
        self.lin = Linear(in_dim, hidden_dim)
        print('Model used: CGN + global attention')
        #setup activation layer
        self.act = ReLU()
                
        # setup message passing layers
        self.conv_list = ModuleList()
        for _ in range(n_conv_layer):
            conv = GCNConv(hidden_dim, hidden_dim, improved=True) 
            self.conv_list.append(conv)

        # batch normalization for fast convergence
        self.batch_norm = BatchNorm1d(hidden_dim)
                
        # global attention layer
        self.gat = Global_Attention(hidden_dim, 2)
        
        # pooling layer
        self.set2set = Set2Set(hidden_dim, 3)
        
        # dropout
        self.dropout = Dropout(dropout_rate)
        
        # Post cgnn layers
        self.mlp = MLP(hidden_dim * 2, n_linear)
    
    def forward(self, data):
        x, edge_index, edge_dist, global_info = \
            data.x, data.edge_index, data.edge_dist, data.global_info

        # pre cgnn
        x = self.act(self.lin(x))
        prev = x
        
        # gatcnn
        for i, conv in enumerate(self.conv_list):
            x = conv(x, edge_index, edge_dist)
            x = self.batch_norm(x)
            #skip connection
            x = torch.add(x, prev)
            x = self.dropout(x)
            prev = x
        
        # weigh each node with global attention
        node_weight = self.gat(x, global_info, data.batch)
        x = x * node_weight
        
        # pooling and dropout
        x = self.set2set(x, data.batch)
        x = self.dropout(x)

        # post cgnn    
        x = self.mlp(x)

        return x

            
