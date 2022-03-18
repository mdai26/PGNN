# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 13:07:36 2022

@author: daimi
"""

from CNNdata import CNNpolygrain, get_train_val_test_loader

# specify number of groups (groups starts from 1 and each group contain 100 data points)
group = 2
Nx = 64; Ny = 64; Nz = 64
# load data
dataset = CNNpolygrain(group, Nx, Ny, Nz)
_,target = dataset[0]
print(target)
# data split
# set random seed
random_seed = 20
# set batch size
batch_size = 1
# set train,validation, testing ratio
train_ratio = 0.8; val_ratio = 0.2
# set flag of using cuda or not.
fcuda = False
# data split
train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset, random_seed, batch_size, train_ratio, val_ratio, fcuda)
