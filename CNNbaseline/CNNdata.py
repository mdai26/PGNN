# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 23:09:27 2022

@author: daimi
"""

# This is the file to load data

from __future__ import print_function
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
from torch.utils.data.sampler import SubsetRandomSampler


class CNNpolygrain(Dataset):
    def __init__(self, group, Nx, Ny, Nz):
        # There are 100 data points in each group
        numdata = 100
        for g in range(1, group+1):
            # read conductivity file first
            foldername = '%d' % g
            condfile = os.path.join(foldername, 'finalconductivity.txt')
            gcond, gbcond, calcond = readcond(condfile, numdata)
            # go through data points
            for n in range(numdata):
                # read structure file
                fstruct = '%s/struct_%d.in' % (foldername, n)
                struct = readstruct(fstruct, Nx, Ny, Nz)
                # read euler angle
                feulerang = '%s/eulerAng_%d.in' % (foldername, n)
                eulerang = readeulerang(feulerang, Nx, Ny, Nz)
                # get the conductivity matrix according to the structure id
                conductivity = getcond(struct,gcond[n,:],gbcond[n,:])
                # combine the conductivity matrix and the euler angle matrix
                gimage = getimage(conductivity, eulerang)
                if (g == 1) and (n == 0):
                    gimagelist, targetlist = [gimage],[calcond[n,:]]
                else:
                    gimagelist, targetlist = np.concatenate((gimagelist, [gimage])), \
                                            np.concatenate((targetlist, [calcond[n,:]]))
                #print("successfully read data %d from group %d" % (n, g), flush = True)
                
        self.gimagelist = np.array(gimagelist)
        self.targetlist = np.array(targetlist)
        
        print('Dataset')
        print('Image matrix shape: ', self.gimagelist.shape)
        print('target conductivity shape: ', self.targetlist.shape)
    
    def __len__(self):
        return len(self.gimagelist)
    
    def __getitem__(self,dataid):
        gimage = self.gimagelist[dataid]
        target = self.targetlist[dataid]
        
        gimage = torch.from_numpy(gimage)
        target = torch.from_numpy(target)
        
        return gimage,target
        


def readcond(condfile,numdata):
    # conductivity from file
    cond = np.loadtxt(condfile)
    gcond = np.copy(cond[:numdata,0:3])
    gbcond = np.copy(cond[:numdata,3:6])
    calcond = np.copy(cond[:numdata,6:9])
    return gcond, gbcond, calcond

def readstruct(fstruct, Nx, Ny, Nz):
    # read structure data from file
    data = np.loadtxt(fstruct, skiprows = 1, dtype = np.int64)
    # get the structure data 
    struct = np.zeros((Nx, Ny, Nz))
    for i in range(len(data)):
        struct[data[i,0]-1,data[i,1]-1,data[i,2]-1] = data[i,3]
    
    return struct

def readeulerang(feulerang, Nx, Ny, Nz):
    # read euler ang data from file
    data = np.loadtxt(feulerang, skiprows = 1)
    # get the euler angle data
    eulerang = np.zeros((Nx, Ny, Nz,3))
    for i in range(len(data)):
        eulerang[int(data[i,0]-1), int(data[i,1]-1), int(data[i,2]-1),0:3] = data[i,3:6]
    
    return eulerang

def getcond(struct,gcond,gbcond):
    Nx, Ny, Nz = struct.shape
    cond = np.zeros((Nx, Ny, Nz, 3))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                if struct[i,j,k] == 1:
                    cond[i,j,k,:] = gcond
                else:
                    cond[i,j,k,:] = gbcond
    
    return cond

def getimage(cond, eulerang):
    gimage = np.concatenate((cond, eulerang), axis = 3)
    return gimage
                    
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
    
    
    
    
        
    
