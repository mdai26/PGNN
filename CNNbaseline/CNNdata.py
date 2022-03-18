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
import pandas as pd
import random
from torch.utils.data.sampler import SubsetRandomSampler

# feature list #
# grain_matrix [max_node, num_feature]
# neighbor_list [max_node, num_edge]
# conduct
# target

class CNNpolygrain(Dataset):
    def __init__(self, num_micro, num_cond, Nx, Ny, Nz):
        
        # read the conductivity and calculated results from "conductivity.csv"
        condfile = 'struct_and_eulerang/conductivity.csv'
        incod, target = readcond(num_micro, num_cond, condfile)
        # go through data points
        for i in range(num_micro):
            # read 3 structure id
            fstruct = 'struct_and_eulerang/struct_%d.in' % i
            struct = readstruct(fstruct, Nx, Ny, Nz)
            # read euler angle
            feulerang = 'struct_and_eulerang/eulerAng_%d.in' % i
            eulerang = readeulerang(feulerang, Nx, Ny, Nz)
            # put data into the final list
            for j in range(num_cond):
                # get the conductivity matrix according to the structure id
                cond = getcond(struct,incod[i,j,:])
                # combine the conductivity matrix and the euler angle matrix
                gimage = getimage(cond, eulerang)
                # concatenate data points
                if (i == 0) and (j == 0):
                    gimagelist, targetlist = [gimage],[target[i,j,:]]
                else:
                    gimagelist, targetlist = np.concatenate((gimagelist, [gimage])), \
                                            np.concatenate((targetlist, [target[i,j,:]]))
                
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
        


def readcond(num_data, num_cond, condfile):
    # read initial conductivity and calculated conductivity from file
    df = pd.read_csv(condfile,index_col=0)
    data = df.to_numpy()
    # specify the array of input conductivity and target
    incod = np.zeros((np.shape(data)[0],num_cond,3))
    target = np.zeros((np.shape(data)[0],num_cond,3))
    for i in range(np.shape(data)[0]):
        for j in range(num_cond):
            # read input conductivity
            incod[i,j,0] = 3.2 * np.power(10, data[i,1 + j * 4])
            incod[i,j,1] = 3.2 * np.power(10, data[i,1 + j * 4])
            incod[i,j,2] = 1.6 * np.power(10, data[i,1 + j * 4])
            # read output conductivity
            target[i,j,0:3] = data[i,(2 + j * 4) : (5 + j * 4)]
            
    return incod, target

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

def getcond(struct,incod):
    Nx, Ny, Nz = struct.shape
    cond = np.zeros((Nx, Ny, Nz, 3))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                if struct[i,j,k] == 1:
                    cond[i,j,k,:] = 3.2e-5, 3.2e-5, 1.6e-5
                else:
                    cond[i,j,k,:] = incod
    
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
    
    
    
    
        
    
