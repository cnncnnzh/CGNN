# -*- coding: utf-8 -*-
"""
Created on Wed May 31 23:34:03 2023
"""

import torch
from torch.nn import Linear, ModuleList, ReLU, Softplus, BatchNorm1d, Dropout
from torch_geometric.utils import softmax
from torch_geometric.nn import global_mean_pool, Set2Set, GCNConv, DiffGroupNorm
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.utils import softmax as tg_softmax
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv  import MessagePassing
from torch_geometric.nn.inits import glorot, zeros

class GATGNN_AGAT_LAYER(MessagePassing):
    def __init__(self, dim, act, dropout_rate, fc_layers=2, **kwargs):
        super(GATGNN_AGAT_LAYER, self).__init__(aggr='add',flow='target_to_source', **kwargs)

        self.act          = act
        self.fc_layers    = fc_layers


        self.dropout_rate = dropout_rate
 
        # FIXED-lines ------------------------------------------------------------
        self.heads             = 4
        self.add_bias          = True
        self.neg_slope         = 0.2

        self.bn1               = nn.BatchNorm1d(self.heads)
        self.W                 = Parameter(torch.Tensor(dim*2,self.heads*dim))
        self.att               = Parameter(torch.Tensor(1,self.heads,2*dim))
        self.dim               = dim

        if self.add_bias  : self.bias = Parameter(torch.Tensor(dim))
        else              : self.register_parameter('bias', None)
        self.reset_parameters()
        # FIXED-lines -------------------------------------------------------------

    def reset_parameters(self):
        glorot(self.W)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x,edge_attr=edge_attr)

    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr): 
        out_i   = torch.cat([x_i,edge_attr],dim=-1)
        out_j   = torch.cat([x_j,edge_attr],dim=-1)
        
        out_i   = self.act(torch.matmul(out_i,self.W))
        out_j   = self.act(torch.matmul(out_j,self.W))
        out_i   = out_i.view(-1, self.heads, self.dim)
        out_j   = out_j.view(-1, self.heads, self.dim)

        alpha   = self.act((torch.cat([out_i, out_j], dim=-1)*self.att).sum(dim=-1))
        alpha   = self.act(self.bn1(alpha))
        alpha   = tg_softmax(alpha,edge_index_i)

        alpha   = F.dropout(alpha, p=self.dropout_rate, training=self.training)
        out_j     = (out_j * alpha.view(-1, self.heads, 1)).transpose(0,1)
        return out_j

    def update(self, aggr_out):
        out = aggr_out.mean(dim=0)
        if self.bias is not None:  out = out + self.bias
        return out
    
class MLP(torch.nn.Module):
    '''
    multilayer perceptron after message passing layers
    '''
    def __init__(self,
                 in_dim,
                 n_linear,
                 act):
        super(MLP, self).__init__()
        self.lin_list = ModuleList()
        for _ in range(n_linear):
            self.lin_list.append(Linear(in_dim, in_dim))
        self.act = getattr(F, act)
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
    def __init__(self, dim, n_layers, act):
        super(Global_Attention, self).__init__()
        
        self.in_layer = torch.nn.Linear(dim + 107, dim)
        
        self.lin_list = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.lin_list.append(torch.nn.Linear(dim, dim))
        
        self.out_layer = torch.nn.Linear(dim, 1)
        self.act = getattr(F, act)        

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
        edge_dim,
        hidden_dim=64,
        n_conv_layer=5,
        n_linear=1,
        dropout_rate=0.2,
        act="softplus"
    ):
        super(SKIP, self).__init__()


        print('Model used: CGN + skip connection + global attention')
        # setup linear layer before gnn
        self.lin_node= Linear(in_dim, hidden_dim)
        self.edge_node= Linear(edge_dim, hidden_dim)

        #setup activation layer
        self.act = self.act = getattr(F, act)
                
        # setup message passing layers
        self.conv_list = ModuleList()
        for _ in range(n_conv_layer):
            # conv = GCNConv(hidden_dim * 2, hidden_dim, improved=True) 
            conv = GATGNN_AGAT_LAYER(hidden_dim, self.act, dropout_rate)
            self.conv_list.append(conv)

        # batch normalization for fast convergence
        self.batch_norm = DiffGroupNorm(hidden_dim, 10)       
        #self.batch_norm = BatchNorm1d(hidden_dim)
                
        # global attention layer
        self.gat = Global_Attention(hidden_dim, 2, act)
        
        # pooling layer
        self.set2set = Set2Set(hidden_dim, 3)
        
        # dropout
        self.dropout = Dropout(dropout_rate)
        
        # Post cgnn layers
        self.mlp = MLP(hidden_dim * 2, n_linear, act)
    
    def forward(self, data):
        x, edge_index, edge_dist, global_info, edge_attr = \
            data.x, data.edge_index, data.edge_dist, data.global_info, data.edge_attr
        
        # pre cgnn
        x = self.act(self.lin_node(x))
        edge_attr = self.edge_node(edge_attr)
        edge_attr = getattr(F, 'leaky_relu')(edge_attr,0.2)
        prev = x
        
        # gatcnn
        for i, conv in enumerate(self.conv_list):
            # x = conv(x, edge_index, edge_dist)
            x = conv(x, edge_index, edge_attr)
            x = self.batch_norm(x)
            #skip connection
            x = torch.add(x,  prev)
            x = self.dropout(x)
            prev = x
        
        # weigh each node with global attention
        node_weight = self.gat(x, global_info, data.batch)
        x = x * node_weight
        
        # pooling and dropout
        x = self.set2set(x, data.batch)
        #x = self.dropout(x)

        # post cgnn    
        x = self.mlp(x)

        return x

            
