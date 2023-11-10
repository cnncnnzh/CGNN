# -*- coding: utf-8 -*-
"""
Created on Wed May 31 23:34:03 2023
"""

import os
import torch
from torch_geometric.data import DataLoader

from cgnn.train import evaluate
from cgnn.data import generate_dataset
from cgnn.utils import write_csv

def predict(
    root_dir,
    data_options,
    loss_func
):
    """
    Use pre-trained model to predict new data and save the results
    """
    
    pseudo_array = []
    target = os.path.join(root_dir, 'targets.csv')
    if not os.path.exists(target):
        for file in os.listdir(root_dir):
            f = os.path.join(root_dir, file)
            if f.endswith('.cif'):
                pre, suf = file.split('.')
                pseudo_array.append([pre, '0'])
        write_csv(pseudo_array, target)        
    
    dataset = generate_dataset(root_dir, data_options)
    loader = DataLoader(dataset, batch_size=1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print('running on gpu')
    if device.type == 'cpu':
        print('running on cpu')
    
    model_dir = os.path.join(root_dir, 'models')
    model_path = os.path.join(model_dir, 'model.pt')
    assert os.path.exists(model_path), 'model not founded'
    model = torch.load(model_path)
    # model.to(device)

    target = os.path.join(root_dir, 'predictions.csv')
    evaluate(model, loader, device, target=target, predict=True)














