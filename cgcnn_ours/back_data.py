# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 21:20:20 2023

@author: zhuhe
"""
import os
import csv
import random
import functools
import numpy as np
import warnings

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pymatgen.core.structure import Structure

from torch_geometric.data import DataLoader, InMemoryDataset, Data

from atom_props import ATOM_PROPS
class Data_parser(Dataset):
    """
    read and parse structure files (for example .cif) target file. Return
    properties and targets in __getitem__() method.     
    """
    
    def __init__(self,
                 root_dir,
                 max_num_nbr=12,
                 radius=8,
                 dmin=0,
                 step=0.2,
                 random_seed=64):
        self.root_dir = root_dir
        self.max_num_nbr = max_num_nbr 
        self.radius = radius
        id_prop_file = os.path.join(self.root_dir, 'targets.csv')
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)
        
    @functools.lru_cache()  # Cache loaded structures
    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx]
        target = torch.Tensor([float(target)])
        # generate pymatgen Structure object for idx 
        crystal = Structure.from_file(os.path.join(self.root_dir, cif_id+'.cif'))
        # add atomic features
        atom_fea = np.vstack([[crystal[i].specie.number]
                              for i in range(len(crystal))])
        atom_fea = torch.Tensor(atom_fea)
        # add neighbors to each atom and sort them based on distance 
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(cif_id))
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr -
                                                     len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id
    
    def __len__(self):
        return len(self.id_prop_data)
