# -*- coding: utf-8 -*-
"""
Created on Wed May 31 23:34:03 2023
"""

import csv

def write_csv(array, target):
    with open(target, 'a', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(array)
        