# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 10:27:44 2022

@author: daimi
"""

import numpy as np
import voronoi
from pyevtk.hl import gridToVTK
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
    grainmark = np.ones((Nx, Ny, Nz))
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
                    grainmark[i,j,k] = 2

    
    return grainmark

def outputvtk(mark, newmark, dataid, Nx, Ny, Nz):
    # specify location
    x = np.zeros((Nx, Ny, Nz))
    y = np.zeros((Nx, Ny, Nz))
    z = np.zeros((Nx, Ny, Nz))
    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                x[i,j,k] = i 
                y[i,j,k] = j 
                z[i,j,k] = k
    # specify filename
    filename = "./mark_and_newmark_%d" % dataid
    gridToVTK(filename, x, y, z, pointData = {"mark": mark, "newmark": newmark})
                        

# set random seed
np.random.seed(20)
# specify the limit of number of grains
lowerlimit = 10
upperlimit = 11
# specify number of microstructures
ndata = 1
# get the number of grain array
ngrain = np.random.randint(lowerlimit, upperlimit, ndata, dtype=int)
# define number of grain thickness
gbwidth = 3
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
    
    outputvtk(mark, grainmark, i, Nx, Ny, Nz)
    
        
        
