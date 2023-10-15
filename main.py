# -*- coding: utf-8 -*-
"""
Created on Wed May 31 23:34:03 2023
"""

import argparse



parser = argparse.ArgumentParser(description='CGNN')

parser.add_argument('--root_dir', required=True)

parser.add_argument('--max_num_nbr', default=12, help='max number of neighbors')
parser.add_argument('--radius', default=8, help='radius in which the neighbor atoms'
                                                    'can be considered')
parser.add_argument('--train_ratio', default=0.8, help='percentage of data '
                                                           'used for training')
parser.add_argument('--val_ratio', default=0.1, help='percentage of data '
                                                           'used for validation')
parser.add_argument('--test_ratio', default=0.1, help='percentage of data '
                                                           'used for test')
def main():






if __name__ == '__main__':
