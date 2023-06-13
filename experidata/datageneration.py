# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 10:27:44 2022

@author: daimi
"""

import numpy as np
import voronoi
import featurecal
import outputfunc
import math

'''
preprocessing: remark the labels
'''

def remark(micro):
    # convert the mark from 1 to N
    unique = np.unique(micro)
    indexdict = {unique[i]:i+1 for i in range(len(unique))}
    newmicro = np.copy(micro)
    for key, value in indexdict.items():
        newmicro[micro==key] = value

    return newmicro, len(unique)

'''
get the adjacency matrix of the microstructure
'''
def getgrainneighbor(ngrain, mark, periodic):
    Nx, Ny, Nz = mark.shape
    # define neighboring matrix
    neighbor = np.zeros((ngrain, ngrain))
    # find neighbors
    # if two grain are neighbors, there should exist grids with mark of the two grains neighbors.  
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                mark0 = mark[i,j,k]
                # get neighbors
                neighbors = voronoi.getneighbor(i,j,k,Nx,Ny,Nz,periodic)
                for n in neighbors:
                    markn = mark[n[0], n[1], n[2]]
                    if markn != mark0:
                        neighbor[int(markn-1), int(mark0-1)] = 1
                        neighbor[int(mark0-1), int(markn-1)] = 1
    return neighbor
'''
remove grain mark with zero area
'''
def rmzerograin(grainmark, neighbor, igrain):
    # define two list, 
    # one for deleted grain marks
    # one for remained grain marks
    delete, remain = [], []
    for n in range(igrain):
        if np.sum(grainmark==n+1) == 0:
            delete.append(n)
        else:
            remain.append(n)
    # delete neighbor first
    neighbor = np.delete(neighbor, delete, axis = 0)
    neighbor = np.delete(neighbor, delete, axis = 1)
    # get the number of grains
    igrain = len(remain)
    # update the grainmark to make it consistent
    remaindict = {remain[i]:int(i+1) for i in range(igrain)}
    gx, gy, gz = grainmark.shape
    for k in range(gz):
        for j in range(gy):
            for i in range(gx):
                temp = grainmark[i, j, k]
                if temp != 0:
                    grainmark[i, j, k] = remaindict[int(temp-1)]
                
    return grainmark, neighbor, igrain


'''
flip the microstructure
'''

def microflip(grainmark):
    Nx, Ny, Nz = grainmark.shape
    grainmark = np.roll(grainmark, int(Nx/2), axis = 0)
    grainmark = np.roll(grainmark, int(Ny/2), axis = 1)
    grainmark = np.roll(grainmark, int(Nz/2), axis = 2)
    
    return grainmark

'''
flip the euler angles
'''
def eulerflip(eulerAng):
    Nx, Ny, Nz, _ = eulerAng.shape
    eulerAng = np.roll(eulerAng, int(Nx/2), axis = 0)
    eulerAng = np.roll(eulerAng, int(Ny/2), axis = 1)
    eulerAng = np.roll(eulerAng, int(Nz/2), axis = 2)

    return eulerAng
    

def geteuler(igrain, grainmark, euler):
    # get the dimension
    gx, gy, gz = grainmark.shape
    # get the list of euler angles
    euleranglist = np.zeros((igrain, 3))
    numbervoxel = np.zeros((igrain,1))
    for i in range(gx):
        for j in range(gy):
            for k in range(gz):
                temp = grainmark[i, j, k]
                if temp != 0:
                    eulertemp = euler[i, j, k, :] / math.pi * 180
                    euleranglist[temp-1, :] += eulertemp
                    numbervoxel[temp-1,0] += 1

    # get corresponding euler angles
    for i in range(igrain):
        euleranglist[i,:] = euleranglist[i,:] / numbervoxel[i,0]

    for i in range(gx):
        for j in range(gy):
            for k in range(gz):
                temp = grainmark[i, j, k]
                # for grain: use the original euler angles
                if temp != 0:
                    euler[i, j, k, :] = euleranglist[temp-1, :]
                # for grain boundary: give random euler angles
                else:
                    euler[i, j, k, 0] = np.random.uniform(0, 360)
                    euler[i, j, k, 1] = np.random.uniform(0, 180)
                    euler[i, j, k, 2] = np.random.uniform(0, 360)

    return euler, euleranglist

def getstructure(grainmark):
    Nx, Ny, Nz = grainmark.shape
    # for grain: the structure id is 1
    structure = np.ones((Nx, Ny, Nz))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                # for grain boundary: the structure id is 2
                if grainmark[i, j, k] == 0:
                    structure[i, j, k] = 2
                    
    return structure
 
                       

# set random seed
random_seed = 10
np.random.seed(random_seed)

# define number of grain thickness
gbwidth = 1
# define dimension of the microstruture
Nx = 70; Ny = 70; Nz = 70
# define use periodic boundary condition or not
periodic = True


# read data
eulerfile = 'EulerAngles.txt'
featurefile = 'FeatureIds.txt'
#microlist, eulerlist = preprocessing.getEBSDdata(eulerfile, featurefile)
microlist, eulerlist = np.load('microlist.npy'), np.load('eulerlist.npy')
print("Finish reading EBSD data", flush = True)
numdata = len(microlist)
ngrainlist = np.zeros((numdata,))

for i in range(numdata):
    print("deal with data %d" % i, flush = True)
    # get the corresponding microstructure and euler angles
    micro, euler = microlist[i], eulerlist[i]
    # get mark and number of grains
    mark, igrain = remark(micro)
    print("finish preprocessing", flush = True)
    # flip the microstructure
    mark = microflip(mark)
    # flip the euler angles
    euler = eulerflip(euler)
    # get neighboring relationship
    neighbor = getgrainneighbor(igrain, mark, periodic)
    # generate grain boundaries
    grainmark = voronoi.addboundary(mark, periodic)
    print("add grain boundaries", flush = True)
    # delete grains with zero area
    grainmark, neighbor, igrain = rmzerograin(grainmark, neighbor, igrain)
    print("finish delete grains with zero area", flush = True)
    # assign euler angle
    eulerang, euleranglist = geteuler(igrain, grainmark, euler)
    # get structure
    structure = getstructure(grainmark)
    print("finish to get euler angle and structure", flush = True)
    # get grain feature
    featurecal.extractfeature(grainmark, euleranglist, igrain, i, Nx, Ny, Nz)
    print("extract grain features", flush = True)
    # save number of grains
    ngrainlist[i] = igrain
    # output euler angles structures and neighbolist
    outputfunc.outputeuler(eulerang, i, Nx, Ny, Nz)
    outputfunc.outputstruct(structure, "struct", i, Nx, Ny, Nz)
    outputfunc.outputneighbor(neighbor, i, Nx, Ny, Nz)
    # output mark
    outputfunc.outputstruct(grainmark, "grainmark", i, Nx, Ny, Nz)
    outputfunc.outputvtk(mark, structure, i, Nx, Ny, Nz)
    print("finish output", flush = True)
    

# save data information to a file
filename = 'information.txt'
with open(filename, 'w') as ofile:
    ofile.write("microstructure shape: (%d, %d, %d)\n" % (Nx, Ny, Nz))
    ofile.write("random seed: %d\n" % random_seed)
    ofile.write("grain boundary widt: %d\n" % gbwidth)
    ofile.write("periodic flag: %s\n" % periodic)
    ofile.write("start the number of grains list\n")

np.savetxt("numgrain.txt", ngrainlist, newline=" ")
        
        

        
        
