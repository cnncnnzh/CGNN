# -*- coding: utf-8 -*-
"""
Created on Wed May 31 23:34:03 2023
"""

from data import Data_parser, data_loader

def main(data_options,
         batch_options,
         train_options,):
    dataset = Data_parser(**data_options)
    train_loader, val_loader, test_loader = data_loader(dataset, **batch_options)
    return train_loader
    
    
    
def train_model(train_loader, model, criterion, optimizer, epoch):
    """
    main function of training model —— load data, build model, main training loop,
    result 
    """
    
    pass



if __name__ == '__main__':
    data_options = {'root_dir':r'D:\Dropbox\Vasp_home\Machine_learning\root_dir_added',
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
    train_options = {}

    train_loader = main(data_options, batch_options, train_options)