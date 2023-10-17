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
        loss = loss_func(torch.squeeze(pred), torch.squeeze(data.y))
        total_loss += loss.item()
        # count += pred.size(0)
        count += 1
    return total_loss/count

def train(
    root_dir,
    data_options,
    batch_options,
    model_options,
    train_options,
):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print('running on gpu')
    if device.type == 'cpu':
        print('running on cpu')
        
    dataset = generate_dataset(root_dir, data_options)
    
    in_dim = dataset.num_features
    train_loader, val_loader, test_loader = data_loader(dataset, batch_options)

    model = GCNN(in_dim, 1, **model_options)
    # model = GCN(dataset)
    
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
    # print([x.y for x in train_loader.dataset])
    for epoch in range(train_options['epochs']):
        model.train()
        
        # train and calculate training error
        epoch_train_loss = 0
        train_count = 0
        for data in train_loader:
            data = data.to(device=device)
            pred = model(data)
            optimizer.zero_grad()
            # print('pred', pred)
            # print('y', torch.squeeze(data.y))
            loss = loss_func(torch.squeeze(pred), torch.squeeze(data.y))
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            train_count += 1
            # train_count += torch.squeeze(data.y).size(0)
        epoch_train_loss /= train_count
        train_losses.append(epoch_train_loss)
        print('training loss after epoch {} is {}'.format(epoch+1, epoch_train_loss))
        
        scheduler.step(epoch_train_loss)
       
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
    
    print('loss for the best model on the test set is {}'.format(best_loss))
    print('loss for the last model on the test set is {}'.format(last_loss))
    
    return train_losses, val_losses, [best_loss, last_loss]
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # root_dir = r'..\tests'
    root_dir = r'D:\Dropbox\Vasp_home\Machine_learning\deeperGATGNN\data\bulk_data\bulk_data_new'
    # root_dir = r'D:\Dropbox\Vasp_home\Machine_learning\neg_freq'
    data_options = {
        'max_num_nbr':12,
        'radius':8,
        'gstart':0,
        'gstop':5.0,
        'gresolution':50,
        'gwidth':0.05,
        'random_seed':128
    }
    batch_options = {
        'train_ratio':0.8,
        'val_ratio':0.1,
        'test_ratio':0.1,
        'batch_size':256,
        'num_workers':1
    }
    model_options = {
        'hidden_dim':64,
        'n_conv_layer':3,
        'n_linear':1,
        'dropout_rate':0.1
    }
    train_options = {
        'epochs':280,
        'optimizer':'Adam',
        'lr':0.01,
        'scheduler':'LinearLR',
        'loss_func':'L1Loss'
    }
    
    train_losses, val_losses, test_loss = train(root_dir, data_options, batch_options, model_options, train_options)
    plt.plot(train_losses)
    plt.plot(val_losses)