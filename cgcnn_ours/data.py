# -*- coding: utf-8 -*-
"""
Created on Wed May 31 23:34:03 2023
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

def data_loader(dataset,
                train_ratio,
                val_ratio,
                test_ratio,
                batch_size=64,
                num_workers=1):
    total_size = len(dataset)
    assert train_ratio + val_ratio + test_ratio == 1
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = int(test_ratio * total_size)
    test_size += total_size - train_size - val_size - test_size
    train_set, val_set, test_set = random_split(dataset,
                                      [train_size, val_size, test_size],
                                      generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers
                              )
    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers
                            )
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_workers
                             )
    
    return train_loader, val_loader, test_loader

## pyg dataset 
class Dataset_pyg(InMemoryDataset):
    def __init__(self,
                 raw_data_path,
                 process_dir='processed',
                 transform=None,
                 pre_transform=None,
                 ):
        self.raw_data_path = raw_data_path
        self.process_dir = process_dir
        super().__init__(raw_data_path, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_dir(self):
        return self.process_dir
    
    @property
    def processed_file_names(self,):
        return ['data.pt']
    
    def process(self, data_options):
        print("Save processed data to: " + self.process_dir)
        
        ## read targets.csv file
        id_prop_file = os.path.join(self.raw_data_path, 'targets.csv')
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            id_prop_data = [row for row in reader]
        np.random.seed(123)
        np.random.shuffle(id_prop_data)
        
        ## read structrues
        data_list = []
        for cif_id, target in id_prop_data:
            data = Data()
            # generate pymatgen Structure object for idx 
            crystal = Structure.from_file(os.path.join(self.raw_data_path, cif_id+'.cif'))
            # add atomic features
            atom_fea = np.vstack([[crystal[i].specie.number]
                                  for i in range(len(crystal))])
            # add neighbors to each atom and sort them based on distance 
            all_nbrs = crystal.get_all_neighbors(data_options['radius'], include_index=True)
            all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
            nbr_fea_idx, nbr_fea = [], []
            for nbr in all_nbrs:
                if len(nbr) < data_options['max_num_nbr']:
                    warnings.warn('{} not find enough neighbors to build graph. '
                                  'If it happens frequently, consider increase '
                                  'radius.'.format(cif_id))
                    nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                       [0] * (data_options['max_num_nbr'] - len(nbr)))
                    nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                                   [data_options['radius'] + 1.] * (data_options['max_num_nbr']
                                                                    - len(nbr)))
                else:
                    nbr_fea_idx.append(list(map(lambda x: x[2],
                                                nbr[:data_options['max_num_nbr']])))
                    nbr_fea.append(list(map(lambda x: x[1],
                                            nbr[:data_options['max_num_nbr']])))
            target = torch.Tensor([float(target)])
            
            edge_index = [[], []]
            for i, neighbor in enumerate(nbr_fea_idx):
                edge_index[0] += len(neighbor) * [i]
                edge_index[1] += neighbor     
            
            edge_dist = np.array(nbr_fea).flatten().reshape(-1,1)
            
            ## global features
            # get crystal system
            system = np.digitize(crystal.get_space_group_info()[0],
                                 np.array([3, 16, 75, 143, 168, 195])) 
            symmetry = np.zeros(7)
            symmetry[system] = 1
            # get global atom indices
            counts = np.unique(atom_fea, return_counts=True)
            global_idx = np.zeros(100)
            global_idx[counts[0].astype(int)] += counts[1] / len(atom_fea)
            
            ## set attributes to data
            # save index of atoms instead of all featrues to save space
            data.x = torch.Tensor(np.vstack([ATOM_PROPS[str(fea[0])] for fea in atom_fea]))
            data.edge_index = torch.LongTensor(edge_index)
            data.edge_attr = torch.Tensor(edge_dist)
            # data.symmetry = symmetry
            # data.global_idx = global_idx
            data_list.append(data)
            
            ## set target values
            data.y = target
        
        return data_list
        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])
       
        

if __name__ == 'main':
    data_options = {'root_dir':r'D:\Dropbox\Vasp_home\Machine_learning\machine-learning\cgcnn_ours\tests',
                    'max_num_nbr':12,
                    'radius':8,
                    'dmin':0,
                    'step':0.2,
                    'random_seed':64}
    batch_options = {'train_ratio':0.7,
                     'val_ratio':0.15,
                     'test_ratio':0.15,
                     'batch_size':32,
                     'num_workers':1}
    dataset = Dataset_pyg(r'D:\Dropbox\Vasp_home\Machine_learning\machine-learning\cgcnn_ours\tests')
    dlist = dataset.process(data_options)
    