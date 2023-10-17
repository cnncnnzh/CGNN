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
from torch_geometric.utils import add_self_loops

from atom_props import ATOM_PROPS
    
def gaussian_converter(nbr_dists, start, stop, resolution, width):
    offset = torch.linspace(start, stop, resolution)
    coeff = -0.5 / ((stop - start) * width) ** 2
    nbr_dists = nbr_dists - offset.view(1, -1)
    return torch.exp(coeff * torch.pow(nbr_dists, 2))
                      

def data_loader(dataset,
                batch_options):
    total_size = len(dataset)
    assert batch_options['train_ratio'] + batch_options['val_ratio'] + batch_options['test_ratio'] == 1
    train_size = int(batch_options['train_ratio'] * total_size)
    val_size = int(batch_options['val_ratio'] * total_size)
    test_size = int(batch_options['test_ratio'] * total_size)
    test_size += total_size - train_size - val_size - test_size
    train_set, val_set, test_set = random_split(dataset,
                                      [train_size, val_size, test_size],
                                      generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(
        train_set,
        batch_size=batch_options['batch_size'],
        shuffle=True,
        num_workers=batch_options['num_workers']
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_options['batch_size'],
        shuffle=True,
        num_workers=batch_options['num_workers']
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_options['batch_size'],
        shuffle=True,
        num_workers=batch_options['num_workers']
    )
    
    return train_loader, val_loader, test_loader


def generate_dataset(root_dir, data_options):
    if os.path.exists(os.path.join(root_dir, 'processed')):
        dataset = Dataset_pyg(root_dir, data_options)
    else:
        Dataset_pyg(root_dir, data_options).process()
        dataset = Dataset_pyg(root_dir, data_options)
    return dataset

class Dataset_pyg(InMemoryDataset):
    def __init__(self,
                 root,
                 data_options,
                 transform=None,
                 pre_transform=None,
                 ):
        self.root = root
        self.process_dir = os.path.join(root, 'processed')
        self.data_options = data_options
        super().__init__(root, transform, pre_transform)
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
    
    def process(self):
        print("Save processed data to: " + self.process_dir)
        
        ## read targets.csv file
        id_prop_file = os.path.join(self.root, 'targets.csv')
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            id_prop_data = [row for row in reader]
        np.random.seed(self.data_options['random_seed'])
        np.random.shuffle(id_prop_data)
        
        ## read structrues
        data_list = []
        count = 0
        for cif_id, target in id_prop_data:
            data = Data()
            
            file = os.path.join(self.root, cif_id+'.cif')
            if not os.path.exists(file):
                warnings.warn('{} in targets.csv not exist. Ignored'.format(cif_id))
                continue
            # generate pymatgen Structure object for idx 
            crystal = Structure.from_file(file)
            # add atomic features
            atom_fea = np.vstack([[crystal[i].specie.number]
                                  for i in range(len(crystal))])
            # add neighbors to each atom and sort them based on distance 
            all_nbrs = crystal.get_all_neighbors(self.data_options['radius'], include_index=True)
            all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
            nbr_fea_idx, nbr_fea = [], []
            for nbr in all_nbrs:
                if len(nbr) < self.data_options['max_num_nbr']:
                    warnings.warn('{} not find enough neighbors to build graph. '
                                  'If it happens frequently, consider increase '
                                  'radius.'.format(cif_id))
                    nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                       [0] * (self.data_options['max_num_nbr'] - len(nbr)))
                    nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                                   [self.data_options['radius'] + 1.] * (self.data_options['max_num_nbr']
                                                                    - len(nbr)))
                else:
                    nbr_fea_idx.append(list(map(lambda x: x[2],
                                                nbr[:self.data_options['max_num_nbr']])))
                    nbr_fea.append(list(map(lambda x: x[1],
                                            nbr[:self.data_options['max_num_nbr']])))
            target = torch.Tensor([float(target)])
            
            edge_index = [[], []]
            for i, neighbor in enumerate(nbr_fea_idx):
                edge_index[0] += len(neighbor) * [i]
                edge_index[1] += neighbor     
            
            edge_dist = np.array(nbr_fea).flatten().reshape(-1,1)
            
            ## global features
            # get crystal system
            system = np.digitize(crystal.get_space_group_info()[1],
                                 np.array([3, 16, 75, 143, 168, 195])) 
            symmetry = np.zeros(7)
            symmetry[system] = 1
            # get global atom indices
            counts = np.unique(atom_fea, return_counts=True)
            global_idx = np.zeros(len(ATOM_PROPS['1']))
            global_idx[counts[0].astype(int)] += counts[1] / len(atom_fea)
            
            edge_index, edge_weight = add_self_loops(
                torch.LongTensor(edge_index), torch.Tensor(edge_dist), fill_value=0
            )
            
            # gaussian smearing of bond distance
            edge_attr = gaussian_converter(edge_weight,
                                              start=self.data_options['gstart'],
                                              stop=self.data_options['gstop'],
                                              resolution=self.data_options['gresolution'],
                                              width=self.data_options['gwidth'])
            
            ## set attributes to data
            # save index of atoms instead of all featrues to save space
            data.x = torch.Tensor(np.vstack([ATOM_PROPS[str(fea[0])] for fea in atom_fea]))
            data.edge_index = edge_index
            data.edge_dist = edge_weight
            data.edge_attr = edge_attr
            data.symmetry = torch.Tensor(symmetry)
            data.global_idx = torch.Tensor(global_idx)
            data_list.append(data)       

            ## set target values
            data.y = target
            # print(data)
            
            count += 1
            if count % 100 == 0:  
                print('processed {} files'.format(count))
            # print(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
       

if __name__ == 'main':
    # root_dir = r'D:\Dropbox\Vasp_home\Machine_learning\machine-learning\cgcnn_ours\tests'
    root_dir = r'D:\Dropbox\Vasp_home\Machine_learning\deeperGATGNN\data\bulk_data\bulk_data_new'
    data_options = {'max_num_nbr':12,
                    'radius':8,
                    'dmin':0,
                    'step':0.2,
                    'random_seed':64}
    batch_options = {'train_ratio':0.7,
                     'val_ratio':0.15,
                     'test_ratio':0.15,
                     'batch_size':32,
                     'num_workers':1}
    dataset = generate_dataset(root_dir, data_options)
    train_loader, val_loader, test_loader = data_loader(dataset, batch_options)