# -*- coding: utf-8 -*-
"""
Created on Wed May 31 23:34:03 2023
"""

import os

from data import generate_dataset, data_loader
from model import GCNN
import torch
from torch import nn
from torch import optim
import copy

def evaluate(
    model,
    loader,
    loss_func,
    device,
):
    total_loss = 0
    count = 0
    for data in loader:
        data = data.to(device=device)
        pred = model(data)
        loss = loss_func(pred, data.y)
        total_loss += loss.item()
        count += pred.size(0)
    return total_loss/count

def train(
    root_dir,
    data_options,
    batch_options,
    model_options,
    train_options,
):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = generate_dataset(root_dir, data_options)
    
    in_dim = dataset.num_features
    train_loader, val_loader, test_loader = data_loader(dataset, batch_options)

    model = GCNN(in_dim, 1, **model_options)
    # load existed model
    saved_model = os.path.join(root_dir, 'saved_model.pt')
    if os.path.exists(saved_model):
        print('resume from the saved model')
        model.load_state_dict(torch.load(saved_model, map_location=device))
    model.to(device=device)
    
    # setup the optimizer
    optimizer = getattr(optim, train_options["optimizer"])(
        model.parameters(),
        lr=train_options['lr'],
    )   
    
    # setup scheduler
    scheduler = getattr(optim.lr_scheduler, train_options["scheduler"])(optimizer)
    
    #setup loss fnction
    loss_func = getattr(nn, train_options['loss_func'])()

    # the main training loop
    train_losses, val_losses = [], []
    best_score = float('inf')
    for epoch in range(train_options['epochs']):
        model.train()
        
        # train and calculate training error
        epoch_train_loss = 0
        train_count = 0
        for data in train_loader:
            data = data.to(device=device)
            pred = model(data)
            optimizer.zero_grad()
            loss = loss_func(pred, data.y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            train_count += pred.size(0)
        train_losses.append(epoch_train_loss/train_count)
        print('training loss after epoch {} is {}'.format(epoch+1, epoch_train_loss/train_count))
        
        # evaluate the model on the validation set
        cur_loss = evaluate(model, val_loader, loss_func, device)
        val_losses.append(cur_loss)
        print('validation loss after epoch {} is {}'.format(epoch+1, cur_loss))
        # save the best model
        if cur_loss < best_score:
            best_model = copy.deepcopy(model)
            torch.save(model.state_dict(), saved_model)
            best_score = cur_loss
    
    # evaluate on the traning set
    best_loss = evaluate(best_model, test_loader, loss_func, device)
    last_loss = evaluate(model, test_loader, loss_func, device)
    
    print('loss for the best model on the best model is {}'.format(best_loss))
    print('loss for the last model on the last model is {}'.format(last_loss))
    
    return train_losses, val_losses, [best_loss, last_loss]
    
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
        'train_ratio':0.8,
        'val_ratio':0.1,
        'test_ratio':0.1,
        'batch_size':3,
        'num_workers':1
    }
    model_options = {
        'hidden_dim':64,
        'n_conv_layer':3,
        'n_linear':1,
        'dropout_rate':0.2
    }
    train_options = {
        'epochs':20,
        'optimizer':'Adam',
        'lr':0.001,
        'dropout':0.2,
        'scheduler':'LinearLR',
        'loss_func':'MSELoss'
    }
    
    train(root_dir, data_options, batch_options, model_options, train_options)