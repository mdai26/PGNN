# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 23:09:27 2022

@author: daimi
"""

# This is the file to load data

from __future__ import print_function
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
from torch.utils.data.sampler import SubsetRandomSampler
        
class polygrainzipdata(Dataset):
    def __init__(self, filename):
        # read data from each dataset filename
        GNNdata = np.load(filename)
        # get different matrices
        nfeature = GNNdata['nfeature']
        neighblist = GNNdata['neighblist']
        efeature = GNNdata['efeature']
        targetlist = GNNdata['targetlist']

        # set it to be features
        self.nfeature = np.array(nfeature)
        self.neighblist = np.array(neighblist)
        self.efeature = np.array(efeature)
        self.targetlist = np.array(targetlist)
        
        print('Dataset', flush = True)
        print('Node feature matrix shape: ', self.nfeature.shape, flush = True)
        print('Neighbor list shape :', self.neighblist.shape, flush = True)
        print('edge feature shape :', self.efeature.shape, flush = True)
        print('target conductivity shape: ', self.targetlist.shape, flush = True)
    
    def __len__(self):
        return len(self.nfeature)
    
    def __getitem__(self,dataid):
        nfeature = self.nfeature[dataid]
        neighblist = self.neighblist[dataid]
        efeature = self.efeature[dataid]
        targetlist = self.targetlist[dataid]
        
        nfeature = torch.from_numpy(nfeature)
        neighblist = torch.from_numpy(neighblist)
        efeature = torch.from_numpy(efeature)
        targetlist = torch.from_numpy(targetlist)
        
        return nfeature, neighblist, efeature, targetlist


def get_train_val_test_loader(train_ds, valid_ds, test_ds, batch_size, pin_memory):
    # get the training, validation and testing dataloader
    train_loader = DataLoader(train_ds, shuffle=True,batch_size = batch_size, pin_memory = pin_memory)
    val_loader = DataLoader(valid_ds, shuffle=True,batch_size = batch_size, pin_memory = pin_memory)
    test_loader = DataLoader(test_ds, shuffle=True,batch_size = batch_size, pin_memory = pin_memory)
    
    return train_loader, val_loader, test_loader
    
    
    
    
        
    
