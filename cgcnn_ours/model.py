# -*- coding: utf-8 -*-
"""
Created on Wed May 31 23:34:03 2023
"""

import torch
import torch.nn as nn

class New_conv(nn.Module):
    
    def __init__(self, atom_fea_len, nbr_fea_len):
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        # TODO will there be problems with the index zero padding?
        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out

class Pooling(nn.Module):
    def __init__(self, atom_fea):
        

class Cgcnn(nn.Module):
    """
    Body of graph neuron network model
    """
    
    def __init__(self,
                 data,
                 atom_fea_len,
                 nbr_fea_len,
                 conv_layers=4
                 ):
        self.linear1 = nn.Linear(atom_fea_len, 64)
        self.new_conv = nn.ModuleList([ConvLayer(atom_fea_len, nbr_fea_len)
                                    for _ in range(conv_layers)])
        self.pooling = nn.AdaptiveAvgPool1d(32)
        self.linear2 = nn.Linear(32, 120)
        self.dropout = nn.Dropout(p=0.4)
        self.relu = nn.ReLU()
        self.conv = nn.Conv1d(48, 1)
    
    def forward():
    