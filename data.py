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
    def __init__(self, group):
        # read data from each group
        for g in range(1, group+1):
            filename = '../../GNNdata_%d.npz' % g
            GNNdata = np.load(filename)
            if g == 1:
                nfeature = GNNdata['nfeature']
                neighblist = GNNdata['neighblist']
                efeature = GNNdata['efeature']
                targetlist = GNNdata['targetlist']
            else:
                nfeature = np.append(nfeature, GNNdata['nfeature'], axis=0)
                neighblist = np.append(neighblist, GNNdata['neighblist'], axis=0)
                efeature = np.append(efeature, GNNdata['efeature'], axis=0)
                targetlist = np.append(targetlist, GNNdata['targetlist'], axis=0)

            print("finish reading data from group %d" % g, flush = True)

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


def get_train_val_test_loader(dataset, random_seed, batch_size, train_ratio, val_ratio, pin_memory):
    # get the number of total data points and shuffle it
    numdata = dataset.__len__()
    dataid = np.arange(numdata)
    random.Random(random_seed).shuffle(dataid)
    # get the number of training, validation and testing data
    train_size = int(train_ratio * numdata)
    val_size = int(val_ratio * numdata)
    test_size = numdata - train_size - val_size
    # get the sampler
    train_sampler = SubsetRandomSampler(dataid[:train_size])
    val_sampler = SubsetRandomSampler(dataid[-(val_size + test_size):-test_size])
    test_sampler = SubsetRandomSampler(dataid[-test_size:])
    # get the training, validation and testing dataloader
    train_loader = DataLoader(dataset, batch_size = batch_size, sampler = train_sampler, pin_memory = pin_memory)
    val_loader = DataLoader(dataset, batch_size = batch_size, sampler = val_sampler, pin_memory = pin_memory)
    test_loader = DataLoader(dataset, batch_size = batch_size, sampler = test_sampler, pin_memory = pin_memory)
    
    return train_loader, val_loader, test_loader
    
    
    
    
        
    
