# -*- coding: utf-8 -*-
"""
Created on Wed May 31 23:34:03 2023
"""

import argparse
import time
import sys
import matplotlib.pyplot as plt
import os

from cgnn import train, predict

parser = argparse.ArgumentParser(description='CGNN')

parser.add_argument(
    '--mode',
    default='train',
    help='mode of running',
    choices=['train', 'predict']
)
parser.add_argument(
    '--root_dir',
    required=True
)
parser.add_argument(
    '--max_num_nbr',
    default='12',
    help='max number of neighbors'
)
parser.add_argument(
    '--radius',
    default='8',
    help='radius in which the neighbor atoms can be considered'
)
parser.add_argument(
    '--gstart',
    default='0',
    help='Minimum interatomic distance in Gaussian Smearing'
)
parser.add_argument(
    '--gstop',
    default='5',
    help='Maximum interatomic distance in Gaussian Smearing'
)
parser.add_argument(
    '--gresolution',
    default='50',
    help='resolution of Gaussian Smearing'
)
parser.add_argument(
    '--gwidth',
    default='0.05',
    help='The spread of Gaussian Smearing'
)
parser.add_argument(
    '--train_ratio',
    default='0.8',
    help='percentage of data used for training'
)
parser.add_argument(
    '--val_ratio',
    default='0.1',
    help='percentage of data used for validation'
)
parser.add_argument(
    '--test_ratio',
    default='0.1',
    help='percentage of data used for test'
)
parser.add_argument(
    '--batch_size',
    default='64',
    help='batch size'
)
parser.add_argument(
    '--hidden_dim',
    default='64',
    help='dimension of the layer after the pre linear layers'
)
parser.add_argument(
    '--n_conv_layer',
    default='3',
    help='number of convolutional layers'
)
parser.add_argument(
    '--n_linear',
    default='1',
    help='number of linear layers after convolutional layers'
)
parser.add_argument(
    '--dropout_rate',
    default='0.1',
    help='dropout rate'
)
parser.add_argument(
    '--epochs',
    default='50',
    help='number of epochs of training'
)
parser.add_argument(
    '--optimizer',
    default='Adam',
    help='choose which optimizer to use'
)
parser.add_argument(
    '--lr',
    default='0.01',
    help='learning rate'
)
parser.add_argument(
    '--scheduler',
    default='LinearLR',
    help='choose which scheduler to use'
)
parser.add_argument(
    '--loss_func',
    default='L1Loss',
    help='choose which loss function to use'
)
parser.add_argument(
    '--workers',
    default='0',
    help='number of subprocesses used to load data'
)


args = parser.parse_args(sys.argv[1:])

def main():
    start = time.time()
    global args
    data_options = {
        'max_num_nbr': int(args.max_num_nbr),
        'radius':float(args.radius),
        'gstart':float(args.gstart),
        'gstop':float(args.gstop),
        'gresolution':int(args.gresolution),
        'gwidth':float(args.gwidth),
        'random_seed':123
    }
    batch_options = {
        'train_ratio':float(args.train_ratio),
        'val_ratio':float(args.val_ratio),
        'test_ratio':float(args.test_ratio),
        'batch_size':int(args.batch_size),
        'num_workers':int(args.workers)
    }
    model_options = {
        'hidden_dim':int(args.hidden_dim),
        'n_conv_layer':int(args.n_conv_layer),
        'n_linear':int(args.n_linear),
        'dropout_rate':float(args.dropout_rate)
    }
    train_options = {
        'epochs':int(args.epochs),
        'optimizer':args.optimizer,
        'lr':float(args.lr),
        'scheduler':args.scheduler,
        'loss_func':args.loss_func
    }
    
    if args.mode == 'train':
        train_losses, val_losses, test_loss = train.train(
            args.root_dir,
            data_options,
            batch_options,
            model_options,
            train_options
            )
        plt.plot(train_losses)
        plt.plot(val_losses)
        
        plt.savefig(os.path.join(args.root_dir, 'epoch_loss.png'))
        
    elif args.mode == 'predict':
        predict.predict(
            args.root_dir,
            data_options,
            args.loss_func
        )

if __name__ == "__main__":
    main()


