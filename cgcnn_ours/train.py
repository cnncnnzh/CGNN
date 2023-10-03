# -*- coding: utf-8 -*-
"""
Created on Wed May 31 23:34:03 2023
"""

import os

from cgcnn_ours.data import generate_dataset, data_loader
from cgcnn_ours.model import GCNN
import torch
from torch import nn
from torch import optim

def train(
    root_dir,
    data_options,
    batch_options,
    train_options,
    model_options
):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = generate_dataset(root_dir, data_options)
    
    in_dim = dataset.num_features
    train_loader, val_loader, test_loader = data_loader(dataset, batch_options)


    model = GCNN(in_dim, 1, **model_options)
    # load existed model
    saved_model = os.path.join(root_dir, 'saved_model.pt')
    if os.path.exists(saved_model):
        model.load_state_dict(torch.load(saved_model, map_location=device))
    model.to(device=device)
    
    # setup the optimizer
    optimizer = getattr(optim, train_options["optimizer"])(
        model.parameters(),
        lr=train_options['lr'],
    )   
    
    # setup scheduler
    scheduler = getattr(optim, train_options["scheduler"])(optimizer)
    
    #setup loss fnction
    loss_func = getattr(nn, train_options['loss_func'])
    
    # the main training loop
    train_losses, train_losses, train_losses = [], [], []
    best_score = float('inf')
    for epoch in range(train_options['batch']):
        model.train()
        epoch_train_loss = 0
        count = 0
        
        # train and calculate training error
        for data in train_loader:
            pred = model(data)
            optimizer.zero_grad()
            loss = loss_func(pred, data.y)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            count += pred.size(0)
        train_losses.append(epoch_train_loss/count) 
    

    # save model
    torch.save(model.state_dict(), saved_model)
    
if __name__ == '__main__':
    
    root_dir = r'D:\Dropbox\Vasp_home\Machine_learning\machine-learning\cgcnn_ours\tests'
    data_options = {
        'max_num_nbr':12,
        'radius':8,
        'dmin':0,
        'step':0.2,
        'random_seed':64
    }
    batch_options = {
        'train_ratio':0.7,
        'val_ratio':0.15,
        'test_ratio':0.15,
        'batch_size':32,
        'num_workers':1
    }
    model_options = {
        'hidden_dim':64,
        'n_conv_layer':3,
        'n_linear':1
    }
    train_options = {
        'batch':20,
        'optimizer':'Adam',
        'lr':0.001,
        'dropout':0.2,
        'scheduler':'LambdaLR',
        'loss_func':'MSELoss'
    }
    
    train(data_options, batch_options, train_options)