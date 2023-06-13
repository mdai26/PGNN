# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 13:42:54 2022

@author: daimi
"""

# This is the code base for voronoi grain growth
# This code provide simple testing microstructure for the labeling grain and finding edge algorithm

import numpy as np
import math

# get the grain center
def getcenter(Nx,Ny,Nz,ngrain):
    # spcecify an empty array 
    gcenter = np.empty((0,3),int)
    while(gcenter.shape[0] < ngrain):
        # get a temporary center location
        ctemp = np.array([[np.random.randint(Nx),np.random.randint(Ny),np.random.randint(Nz)]])
        # append new point to the list
        gcenter = np.append(gcenter, ctemp, axis = 0)
        # get unique location
        gcenter = np.unique(gcenter, axis = 0)
    
    return gcenter

# randomly walk the grain center according to the direction
def walk(i,j,k,direction,Nx,Ny,Nz):
    if direction == 'x+':
        i = i + 1
        if i >= Nx:
            i = i - Nx
    elif direction == 'x-':
        i = i - 1
        if i < 0:
            i = i + Nx
    elif direction == 'y+':
        j = j + 1
        if j >= Nx:
            j = j - Ny
    elif direction == 'y-':
        j = j - 1
        if j < 0:
            j = j + Ny
    elif direction == 'z+':
        k = k + 1
        if k >= Nz:
            k = k - Nz
    elif direction == 'z-':
        k = k - 1
        if k < 0:
            k = k + Nz
            
    return i, j, k

def randomwalk(gcenter, nrwalk, Nx, Ny, Nz):
    # figure out the possible direction according the system size
    direction = []
    if Nx != 1:
        direction.extend(['x+','x-'])
    if Ny != 1:
        direction.extend(['y+','y-'])
    if Nz != 1:
        direction.extend(['z+','z-'])
    # get number of directions
    ndir = len(direction)
    # do random walks
    for gc in range(gcenter.shape[0]):
        for num in range(nrwalk):
            # get the random move
            move = direction[np.random.randint(ndir)]
            # perform move
            gcenter[gc,0], gcenter[gc,1], gcenter[gc,2] = walk(gcenter[gc,0], gcenter[gc,1], gcenter[gc,2], move, Nx, Ny, Nz)
            
    return gcenter

# generate voronoi tesselation
def voronoi(gcenter, Nx, Ny, Nz,periodic):
    # specify the mark array
    mark = np.zeros((Nx, Ny, Nz))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                mindist = float('inf')
                # calculate the minimum distance between one mark
                for gc in range(gcenter.shape[0]):
                    gx, gy, gz = gcenter[gc,:]
                    # perform periodic boundary condition
                    if periodic:
                        distx = min(abs(gx-i), Nx - abs(gx-i))
                        disty = min(abs(gy-j), Ny - abs(gy-j))
                        distz = min(abs(gz-k), Nz - abs(gz-k))
                    else:
                        distx = abs(gx-i)
                        disty = abs(gy-j)
                        distz = abs(gz-k)
                    dist = math.sqrt(distx**2 + disty**2 + distz**2)
                    if dist < mindist:
                        mindist = dist
                        mark[i,j,k] = gc + 1
    
    return mark

# find neighbor of a specific point 
def getneighbor(i,j,k,Nx,Ny,Nz,periodic):
    neighbor = []
    if Nx != 1:
        'x-'
        il = i - 1
        if il > 0:
            neighbor.append([il,j,k])
        elif il < 0 and periodic:
            il = il + Nx
            neighbor.append([il,j,k])
        'x+'
        ir = i + 1
        if ir <= Nx - 1:
            neighbor.append([ir,j,k])
        elif ir > Nx - 1 and periodic:
            ir = ir - Nx
            neighbor.append([ir,j,k])
    if Ny != 1:
        'y-'
        jl = j - 1
        if jl > 0:
            neighbor.append([i,jl,k])
        elif jl < 0 and periodic:
            jl = jl + Ny
            neighbor.append([i,jl,k])
        'y+'
        jr = j + 1
        if jr <= Nx -1:
            neighbor.append([i,jr,k])
        elif jr > Nx - 1 and periodic:
            jr = jr - Ny
            neighbor.append([i,jr,k])
    if Nz != 1:
        'z+'
        kl = k - 1
        if kl > 0:
            neighbor.append([i,j,kl])
        elif kl<0 and periodic:
            kl = kl + Nz
            neighbor.append([i,j,kl])
        'z-'
        kr = k + 1
        if kr <= Nz - 1:
            neighbor.append([i,j,kr])
        elif kr > Nz - 1 and periodic:
            kr = kr - Nz
            neighbor.append([i,j,kr])
    return neighbor
    

def addboundary(mark, periodic):
    Nx, Ny, Nz = mark.shape
    grainmark = np.copy(mark)
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                mark0 = mark[i,j,k]
                # get neighbors
                neighbors = getneighbor(i,j,k,Nx,Ny,Nz,periodic)
                for n in neighbors:
                    if mark[n[0],n[1],n[2]] != mark0:
                        grainmark[i,j,k] = 0
                        break
    
    return grainmark
                        
# # specify dimension
# Nx = 64; Ny = 64; Nz = 1
# # specify number of grain
# ngrain = 10
# # specify random seed
# random.seed(30)
# np.random.seed(30)
# # get grain center
# gcenter = getcenter(Nx,Ny,Nz,ngrain)
# # perform random walk
# nrwalk = 5
# if nrwalk != 0:
    # gcenter = randomwalk(gcenter, nrwalk, Nx, Ny, Nz)
# # perform voronoi concellation
# mark = voronoi(gcenter, Nx, Ny, Nz, periodic=False)
# # add boudaries
# grainmark = addboundary(mark, periodic=False)
# # plot the voronoi tesselation
# fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20,10))
# ax1.imshow(mark[:,:,0].T,vmin=1, vmax = gcenter.shape[0]+1,cmap='hsv',origin='lower')
# ax1.scatter(gcenter[:,0],gcenter[:,1],c='k',s=50)
# ax1.axis('square')
# ax1.set_xlim([0,64])
# ax1.set_ylim([0,64])
# ax1.set_xlabel('x')
# ax1.set_ylabel('y')
# ax2.imshow(grainmark[:,:,0].T, origin='lower')