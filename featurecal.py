# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 08:40:26 2022

@author: daimi
"""

import numpy as np
import os
import math

def mean_var(meanold,varold,n,point):
    meannew = ((n-1)*meanold + point)/n
    varnew = (n-1)*varold/n + (n-1)*(meanold-point)**2/n**2
    return meannew, varnew

def findpos(newmark, ngrain, Nx, Ny, Nz):
    # define the average and variance of position
    avepos = np.zeros((ngrain,3))
    varpos = np.zeros((ngrain,3))
    # define number of grids
    gsize = np.zeros((ngrain,1))
    # start to calculate the position
    cubesize = np.array([Nx, Ny, Nz])
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                gid = int(newmark[i,j,k])
                # for grain 
                if gid != 0:
                    gsize[gid-1, 0] += 1                    
                    point = np.array([i,j,k])
                    for dimen in range(3):
                        ave = np.zeros(3); var = np.zeros(3)
                        # get the average and variance of point with/without periodic treatment
                        ave[0], var[0] = mean_var(avepos[gid-1,dimen],varpos[gid-1,dimen],gsize[gid-1,0],point[dimen])
                        ave[1], var[1] = mean_var(avepos[gid-1,dimen],varpos[gid-1,dimen],gsize[gid-1,0],point[dimen]-cubesize[dimen])
                        ave[2], var[2] = mean_var(avepos[gid-1,dimen],varpos[gid-1,dimen],gsize[gid-1,0],point[dimen]+cubesize[dimen])
                        # choose one with smallest variance
                        minsitu = np.where(var == np.amin(var))[0]
                        # add a check
                        if len(minsitu) > 1: minsitu = minsitu[0]
                        avepos[gid-1,dimen] = ave[minsitu]; varpos[gid-1,dimen] = var[minsitu]
    
    # final check of average position
    for number in range(ngrain):
        for dimen in range(3):
            if avepos[number,dimen] < 0:
                avepos[number,dimen] = avepos[number,dimen] + cubesize[dimen]
            if avepos[number,dimen] >= cubesize[dimen]:
                avepos[number,dimen] = avepos[number,dimen] - cubesize[dimen]
    
    return avepos, gsize

def findedge(avepos,gsize,ngrain, Nx, Ny, Nz):
    edgelist = []
    # calculate equivalent radius
    radius = np.zeros((ngrain,1))
    if (Nx > 1) and (Ny > 1) and (Nz > 1):
        radius[:,0] = (3*gsize[:,0]/(4*math.pi))**(1/3.)
    else:
        radius[:,0] = (gsize[:,0]/math.pi)**(1/2)
    
    cubesize = np.array([Nx, Ny, Nz])    
    for i in range(ngrain):
        for j in range(i+1,ngrain):
            diff = np.zeros(3)
            for dimen in range(3):
                diff[dimen] = min(abs(avepos[i,dimen]-avepos[j,dimen]),cubesize[dimen]-abs(avepos[i,dimen]-avepos[j,dimen]))
            
            if np.linalg.norm(diff) <= radius[i,0] + radius[j,0] +1 :
                edgelist.append([i,j])
                
    return edgelist


    


def extractfeature(newmark, eulerang, ngrain, path, dataid, Nx, Ny, Nz):
    # find position and gsize
    avepos, gsize = findpos(newmark, ngrain, Nx, Ny, Nz)
    # get average euler angles
    geuler = np.zeros((ngrain,3))
    
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                gid = int(newmark[i,j,k])
                # for grain 
                if gid != 0:
                    # calculate sum of euler angels
                    geuler[gid-1,0] += eulerang[i,j,k,0]
                    geuler[gid-1,1] += eulerang[i,j,k,1]
                    geuler[gid-1,2] += eulerang[i,j,k,2]
    
    # do average
    for i in range(ngrain):
        geuler[i,:] = geuler[i,:]/gsize[i,0]
        
    # output
    filename = 'feature_%d.txt' % dataid
    with open(os.path.join(path,filename),'w') as ofile:
        ofile.write("%8s%8s%8s%8s%8s%8s%8s\n" \
                    % ('aveposx','aveposy','aveposz',\
                       'gsize',\
                       'geuler1','geuler2','geuler3'))
        for i in range(ngrain):
            ofile.write("%8.2f%8.2f%8.2f%8d%8.2f%8.2f%8.2f\n" \
                        % (avepos[i,0], avepos[i,1], avepos[i,2],\
                           gsize[i,0],\
                           geuler[i,0], geuler[i,1], geuler[i,2]))
    
    # get the edge between grains
    edgelist = findedge(avepos, gsize, ngrain, Nx, Ny, Nz)
    
    # output edges
    filename = 'edge_%d.txt' % dataid
    with open(os.path.join(path,filename),'w') as ofile:
        for edge in edgelist:
            ofile.write("%8d%8d\n" % (edge[0],edge[1])) 
