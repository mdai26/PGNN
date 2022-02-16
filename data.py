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

class polygrainDS(Dataset):
    def __init__(self, num_micro, max_node, num_cond):
        
        # read the conductivity and calculated results from "conductivity.csv"
        condfile = 'data/conductivity.csv'
        incod, target = readcond(num_micro, num_cond, condfile)
        # specify the final list
#        nfeature, neighlist, efeature = [], [], []
#        incondlist, targetlist = [], []
        # go through data points
        for i in range(num_micro):
            # read node feature
            filenode = 'data/feature_%d.txt' % i
            nodefeature = readnode(filenode,max_node)
            # read neighborlist, edgeid and edgefeature
            fileedge = 'data/edge_%d.txt' % i
            neighbor, edgefeature = readedge(fileedge, max_node)
            # put data into the final list
            for j in range(num_cond):
                if (i == 0) and (j == 0):
                    nfeature, neighblist, efeature, incondlist, targetlist = [nodefeature], [neighbor], [edgefeature], [incod[i,j,:]], [target[i,j,:]]
                else:
                    nfeature, neighblist, efeature, incondlist, targetlist = np.concatenate((nfeature, [nodefeature])), \
                                                                             np.concatenate((neighblist, [neighbor])), \
                                                                             np.concatenate((efeature, [edgefeature])), \
                                                                             np.concatenate((incondlist, [incod[i,j,:]])),\
                                                                             np.concatenate((targetlist, [target[i,j,:]]))
                
        self.nfeature = np.array(nfeature)
        self.neighblist = np.array(neighblist)
        self.efeature = np.array(efeature)
        self.incondlist = np.array(incondlist)
        self.targetlist = np.array(targetlist)
        
        print('Dataset')
        print('Node feature matrix shape: ', self.nfeature.shape)
        print('Neighbor list shape :', self.neighblist.shape)
        print('edge feature shape :', self.efeature.shape)
        print('input conductivity shape :', self.incondlist.shape)
        print('target conductivity shape: ', self.targetlist.shape)
    
    def __len__(self):
        return len(self.efeature)
    
    def __getitem__(self,dataid):
        nfeature = self.nfeature[dataid]
        neighblist = self.neighblist[dataid]
        efeature = self.efeature[dataid]
        incondlist = self.incondlist[dataid]
        targetlist = self.targetlist[dataid]
        
        nfeature = torch.from_numpy(nfeature)
        neighblist = torch.from_numpy(neighblist)
        efeature = torch.from_numpy(efeature)
        incondlist = torch.from_numpy(incondlist)
        targetlist = torch.from_numpy(targetlist)
        
        return nfeature, neighblist, efeature, incondlist, targetlist
        


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

def readnode(filenode, max_node):
    # read node data from file
    data = np.loadtxt(filenode, skiprows = 1)
    # put node feature in the numpy matrix with correct shape
    fea_node = np.zeros((max_node, np.shape(data)[1]))
    fea_node[:np.shape(data)[0], :np.shape(data)[1]] = data
    
    return fea_node

def readedge(fileedge, max_node):
    # read edge data from file
    data = np.loadtxt(fileedge,dtype=int)
    # neighborlist: document the neighbors of each node
    # data type: list of lists
    neighborlist = np.zeros((max_node, max_node))
    # edge feature: the feature of each edge
    # data type: list
    edgefeature = np.zeros((max_node, max_node, 3))
    # initial value of edge id is 0
    for i in range(len(data)):
        neighborlist[data[i,0],data[i,1]] = 1
        edgefeature[data[i,0], data[i,1],:] = [3.2e-5, 3.2e-5, 1.6e-5]
        neighborlist[data[i,1],data[i,0]] = 1
        edgefeature[data[i,1], data[i,0],:] = [3.2e-5, 3.2e-5, 1.6e-5]
        
    return neighborlist, edgefeature

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
    
    
    
    
        
    
