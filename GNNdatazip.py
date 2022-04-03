# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 15:32:33 2022

@author: daimi
"""

import numpy as np
import argparse
import os


def loaddata(group, max_node, numdata, upperlimit, lowerlimit):
    # specify grain boundary thickness
    gbwidth = 1
    # read data from specific group
    # read conductivity file first
    foldername = '%d' % group
    condfile = os.path.join(foldername, 'finalconductivity.txt')
    gcond, gbcond, calcond = readcond(condfile, numdata, upperlimit, lowerlimit)
    # go through each data in the group
    for n in range(numdata):
        # read node feature
        filenode = '%d/feature_%d.txt' % (group, n)
        nodefeature = readnode(filenode,max_node,gcond[n,:])
        # read neighbor and edgefeature
        fileneighbor = '%d/neighbor_%d.txt' % (group, n)
        neighbor, edgefeature = readneighbor(fileneighbor, max_node, gbcond[n,:], gbwidth)
        # put data into the final list
        if n == 0:
            nfeature, neighblist, efeature, targetlist = [nodefeature], [neighbor], [edgefeature], [calcond[n,:]]
        else:
            nfeature, neighblist, efeature, targetlist = np.concatenate((nfeature, [nodefeature])), \
                                                                             np.concatenate((neighblist, [neighbor])), \
                                                                             np.concatenate((efeature, [edgefeature])), \
                                                                             np.concatenate((targetlist, [calcond[n,:]]))
    
    nfeature = np.array(nfeature)
    neighblist = np.array(neighblist)
    efeature = np.array(efeature)
    targetlist = np.array(targetlist)
        
    print('Dataset', flush = True)
    print('Node feature matrix shape: ', nfeature.shape, flush = True)
    print('Neighbor list shape :', neighblist.shape, flush = True)
    print('edge feature shape :', efeature.shape, flush = True)
    print('target conductivity shape: ', targetlist.shape, flush = True)
    
    return nfeature, neighblist, efeature, targetlist
        

def normcond(cond, upperlimit, lowerlimit):
    # normalization
    #print('cond',cond[0,:])
    cond = np.log10(cond)
    #print('cond',cond[0,:])
    cond = (cond - lowerlimit) / (upperlimit - lowerlimit)
    #print('cond',cond[0,:])
    return cond


def readcond(condfile,numdata, upperlimit, lowerlimit):
    # conductivity from file
    cond = normcond(np.loadtxt(condfile), upperlimit, lowerlimit)
    gcond = np.copy(cond[:numdata,0:3])
    gbcond = np.copy(cond[:numdata,3:6])
    calcond = np.copy(cond[:numdata,6:9])
    return gcond, gbcond, calcond


def readnode(filenode, max_node, gcond):
    # read node data from file
    data = np.loadtxt(filenode, skiprows = 1)
    # data normalization
    # data values: three positions, grains size, three euler angles
    #print('feature',data[0,:])
    normvector = np.array([[64, 64, 64, 64*64*64, 360, 180, 360]])
    data = data / normvector
    #print('feature',data[0,:])
    # put node feature in the numpy matrix with correct shape
    fea_node = np.zeros((max_node, int(np.shape(data)[1]+3)))
    #print(max_node)
    #print(fea_node.shape)
    #print(data.shape)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type = int, default = 1)
    parser.add_argument('--maxnode', type = int, default = 400)
    parser.add_argument('--numdata', type = int, default = 100)
    parser.add_argument('--upperlimit', type = int, default = -2)
    parser.add_argument('--lowerlimit',type = int, default = -8)
    given_args = parser.parse_args()
    group = given_args.group
    maxnode = given_args.maxnode
    numdata = given_args.numdata
    upperlimit = given_args.upperlimit
    lowerlimit = given_args.lowerlimit

    nfeature, neighblist, efeature, targetlist = loaddata(group, maxnode, numdata, upperlimit, lowerlimit)
    ofilename = 'GNNdata_%d.npz' % group
    np.savez_compressed(ofilename, nfeature = nfeature, neighblist = neighblist, efeature = efeature, targetlist = targetlist)

    GNNdata = np.load(ofilename)
    nfeature = GNNdata['nfeature']
    neighblist = GNNdata['neighblist']
    efeature = GNNdata['efeature']
    targetlist = GNNdata['targetlist']
    print('Dataset', flush = True)
    print('Node feature matrix shape: ', nfeature.shape, flush = True)
    print('Neighbor list shape :', neighblist.shape, flush = True)
    print('edge feature shape :', efeature.shape, flush = True)
    print('target conductivity shape: ', targetlist.shape, flush = True)

