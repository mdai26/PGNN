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
import os
from torch.utils.data.sampler import SubsetRandomSampler

# feature list #
# grain_matrix [max_node, num_feature]
# neighbor_list [max_node, num_edge]
# conduct
# target

class polygrainDS(Dataset):
    def __init__(self, group, max_node):
        # There are 100 data points in each group
        numdata = 100
        # specify grain boundary thickness
        gbwidth = 1
        # read data from each group
        for g in range(1, group+1):
            # read conductivity file first
            foldername = '%d' % g
            condfile = os.path.join(foldername, 'finalconductivity.txt')
            gcond, gbcond, calcond = readcond(condfile, numdata)
            # go through each data in the group
            for n in range(numdata):
                # read node feature
                filenode = '%d/feature_%d.txt' % (g, n)
                nodefeature = readnode(filenode,max_node,gcond[n,:])
                # read neighbor and edgefeature
                fileneighbor = '%d/neighbor_%d.txt' % (g, n)
                neighbor, edgefeature = readneighbor(fileneighbor, max_node, gbcond[n,:], gbwidth)
                # put data into the final list
                if (g == 1) and (n == 0):
                    nfeature, neighblist, efeature, targetlist = [nodefeature], [neighbor], [edgefeature], [calcond[n,:]]
                else:
                    nfeature, neighblist, efeature, targetlist = np.concatenate((nfeature, [nodefeature])), \
                                                                             np.concatenate((neighblist, [neighbor])), \
                                                                             np.concatenate((efeature, [edgefeature])), \
                                                                             np.concatenate((targetlist, [calcond[n,:]]))
        
        targetlist = normalize(targetlist)

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
        


def readcond(condfile,numdata):
    # conductivity from file
    cond = np.loadtxt(condfile)
    gcond = np.copy(cond[:numdata,0:3])
    gbcond = np.copy(cond[:numdata,3:6])
    calcond = np.copy(cond[:numdata,6:9])
    return gcond, gbcond, calcond


def readnode(filenode, max_node, gcond):
    # read node data from file
    data = np.loadtxt(filenode, skiprows = 1)
    # put node feature in the numpy matrix with correct shape
    fea_node = np.zeros((max_node, int(np.shape(data)[1]+3)))
    fea_node[:np.shape(data)[0], :np.shape(data)[1]] = data
    fea_node[:np.shape(data)[0], np.shape(data)[1]:] = gcond
    
    return fea_node

def readneighbor(fileneighbor, max_node, gbcond, gbwidth):
    # read neighbor data from file
    data = np.loadtxt(fileneighbor,dtype=int)
    # specify the size of neighbor
    neighbor = np.zeros((max_node, max_node))
    # specify the size of edge features (3 conductivity and 1 thickness)
    edgefeature = np.zeros((max_node, max_node, 4))
    # put data into neighbor
    neighbor[:np.shape(data)[0], :np.shape(data)[1]] = data
    # put edge feature into data
    for i in range(np.shape(data)[0]):
        for j in range(np.shape(data)[1]):
            if neighbor[i,j] == 1:
                edgefeature[i,j,:3] = gbcond
                edgefeature[i,j, 3] = gbwidth

    return neighbor, edgefeature

def normalize(targetlist):
    t_mean = np.mean(targetlist)
    t_std = np.std(targetlist)
    targetlist = (targetlist - t_mean) / t_std
    # save norm
    norm = np.array([t_mean, t_std])
    np.savez_compressed('norm.npz', norm = norm)

    return targetlist

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
    
    
    
    
        
    
