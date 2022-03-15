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


def getvoronoi(ngrain, nrwalk, periodic, Nx, Ny, Nz):
    gcenter = voronoi.getcenter(Nx,Ny,Nz,ngrain)
    if nrwalk != 0:
        gcenter = voronoi.randomwalk(gcenter, nrwalk, Nx, Ny, Nz)
    mark = voronoi.voronoi(gcenter, Nx, Ny, Nz, periodic=True)
    
    return mark, gcenter

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

def caldist(i, j, k, center, Nx, Ny, Nz):
    gx, gy, gz = center
    # calculate the 
    if periodic:
        distx = min(abs(gx-i), Nx - abs(gx-i))
        disty = min(abs(gy-j), Ny - abs(gy-j))
        distz = min(abs(gz-k), Nz - abs(gz-k))
    else:
        distx = abs(gx-i)
        disty = abs(gy-j)
        distz = abs(gz-k)
    dist = math.sqrt(distx**2 + disty**2 + distz**2)
    
    return dist


def addboundarywwidth(ngrain, gcenter, neighbor, mark, gbwidth):
    Nx, Ny, Nz = mark.shape
    grainmark = np.copy(mark)
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                mark0 = mark[i,j,k]
                gcenter0 = gcenter[int(mark0-1)]
                difdist0 = caldist(i, j, k, gcenter0, Nx, Ny, Nz)
                neighbor0 = neighbor[int(mark0-1),:]
                neighbor0 = np.where(neighbor0 !=0 )[0]
                difdist = np.zeros(len(neighbor0))
                for num in range(len(neighbor0)):
                    n = neighbor0[num]
                    difdist[num] = caldist(i, j, k, gcenter[n], Nx, Ny, Nz)
                difdist = np.sort(difdist)
                if abs(difdist[0] - difdist0) <= gbwidth:
                    grainmark[i,j,k] = 0

    
    return grainmark

def geteuler(igrain, grainmark):
    # get random euler angles of each grain
    alphalist = np.random.uniform(0, 360, igrain)
    betalist = np.random.uniform(0, 180, igrain)
    gammalist = np.random.uniform(0, 360, igrain)
    # get size of the euler angle array
    Nx, Ny, Nz = grainmark.shape
    eulerang = np.zeros((Nx, Ny, Nz, 3))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                gmark0 = grainmark[i, j, k]
                # for grain: assign euler angles based on the generated list
                if gmark0 != 0:
                    eulerang[i, j, k, 0] = alphalist[int(gmark0 - 1)]
                    eulerang[i, j, k, 1] = betalist[int(gmark0 - 1)]
                    eulerang[i, j, k, 2] = gammalist[int(gmark0 - 1)]
                # for grain boundary: give random euler angles
                else:
                    eulerang[i, j, k, 0] = np.random.uniform(0, 360)
                    eulerang[i, j, k, 1] = np.random.uniform(0, 180)
                    eulerang[i, j, k, 2] = np.random.uniform(0, 360)
    
    euleranglist = np.concatenate((alphalist.reshape(-1,1), betalist.reshape(-1,1), gammalist.reshape(-1,1)), axis = 1)
    return eulerang, euleranglist

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
random_seed = 20
np.random.seed(random_seed)
# specify the limit of number of grains
lowerlimit = 10
upperlimit = 20
# specify number of microstructures
ndata = 3
# get the number of grain array
ngrain = np.random.randint(lowerlimit, upperlimit, ndata, dtype=int)
# define number of grain thickness
gbwidth = 1
# define dimension of the microstruture
Nx = 64; Ny = 64; Nz = 64
# define randome walk for grain center
nrwalk = 10;
# define use periodic boundary condition or not
periodic = True



for i in range(ndata):
    igrain = ngrain[i]
    # get the grain mark
    mark, gcenter = getvoronoi(igrain, nrwalk, periodic, Nx, Ny, Nz)
    # get neighboring relationship
    neighbor = getgrainneighbor(igrain, mark, periodic)
    # generate grain boundaries
    if gbwidth == 1:
        grainmark = voronoi.addboundary(mark, periodic)
    else:
        grainmark = addboundarywwidth(igrain, gcenter, neighbor, mark, gbwidth)
    # assign euler angle
    eulerang, euleranglist = geteuler(igrain, grainmark)
    # get structure
    structure = getstructure(grainmark)
    # get grain feature
    featurecal.extractfeature(grainmark, euleranglist, igrain, i, Nx, Ny, Nz)
    # output euler angles structures and neighbolist
    outputfunc.outputeuler(eulerang, i, Nx, Ny, Nz)
    outputfunc.outputstruct(structure, "struct", i, Nx, Ny, Nz)
    outputfunc.outputneighbor(neighbor, i, Nx, Ny, Nz)
    # output mark
    outputfunc.outputstruct(grainmark, "grainmark", i, Nx, Ny, Nz)
    outputfunc.outputvtk(mark, structure, i, Nx, Ny, Nz)
    

# save data information to a file
filename = 'information.txt'
with open(filename, 'w') as ofile:
    ofile.write("microstructure shape: (%d, %d, %d)\n" % (Nx, Ny, Nz))
    ofile.write("random seed: %d\n" % random_seed)
    ofile.write("lower limit: %d, upper limit: %d\n" % (lowerlimit, upperlimit))
    ofile.write("grain boundary widt: %d\n" % gbwidth)
    ofile.write("random walk steps: %d\n" % gbwidth)
    ofile.write("periodic flag: %s\n" % periodic)
    ofile.write("start the number of grains list\n")
    for i in range(ndata):
        ofile.write("%d\n" % ngrain[i])
        
        

        
        
