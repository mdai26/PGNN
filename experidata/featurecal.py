# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 19:36:16 2022

@author: daimi
"""

import numpy as np
import os

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


def extractfeature(newmark, euleranglist, ngrain, dataid, Nx, Ny, Nz):
    # find position and gsize
    avepos, gsize = findpos(newmark, ngrain, Nx, Ny, Nz)
    # get average euler angles
        
    # output
    filename = 'feature_%d.txt' % dataid
    with open(filename,'w') as ofile:
        ofile.write("%8s%8s%8s%8s%8s%8s%8s\n" \
                    % ('aveposx','aveposy','aveposz',\
                       'gsize',\
                       'geuler1','geuler2','geuler3'))
        for i in range(ngrain):
            ofile.write("%8.2f%8.2f%8.2f%8d%8.2f%8.2f%8.2f\n" \
                        % (avepos[i,0], avepos[i,1], avepos[i,2],\
                           gsize[i,0],\
                           euleranglist[i,0], euleranglist[i,1], euleranglist[i,2]))
