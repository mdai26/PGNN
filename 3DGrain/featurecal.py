# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 08:40:26 2022

@author: daimi
"""

import numpy as np
import os
from ngrain import findneigbhors

def findedge(i,j,k,Nx,Ny,Nz,newmark):
    edge = set()
    visited = np.ones((Nx,Ny,Nz))
    l_layer = [[i,j,k]]
    while len(edge) < 2:
        for element in l_layer:
            visited[element[0],element[1],element[2]] = 0
            if newmark[element[0],element[1],element[2]] != 0:
                edge.add(newmark[element[0],element[1],element[2]])
                if len(edge) == 2:
                    return edge
        nl_layer = []
        for element in l_layer:
            posspoints = findneigbhors(element[0],element[1],element[2],Nx,Ny,Nz)
            # check if six neighbors is visited or not
            for point in posspoints:
                if visited[point[0],point[1],point[2]] == 1:
                    nl_layer.append(point)
        
        l_layer = nl_layer
    


def extractfeature(newmark, eulerang, ngrain, path, dataid, Nx, Ny, Nz):
    # average position
    avepos = np.zeros((ngrain,3))
    # grain size
    gsize = np.zeros((ngrain,1))
    # euler angle
    geuler = np.zeros((ngrain,3))
    
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                gid = int(newmark[i,j,k])
                # for grain 
                if gid != 0:
                    # calculate the sum of positions
                    avepos[gid-1,0] += i
                    avepos[gid-1,1] += j
                    avepos[gid-1,2] += k
                    # calculate the grain size
                    gsize[gid-1,0] += 1
                    # calculate sum of euler angels
                    geuler[gid-1,0] += eulerang[i,j,k,0]
                    geuler[gid-1,1] += eulerang[i,j,k,1]
                    geuler[gid-1,2] += eulerang[i,j,k,2]
    
    # do average
    for i in range(ngrain):
        avepos[i,:] = avepos[i,:]/gsize[i,0]
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
    edgelist = []
    gblist = []
    # get grain boundary elements
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                if newmark[i,j,k]==0:
                    gblist.append([i,j,k])
    # calculate the edges
    for gb in gblist:
        edgelist.append(findedge(gb[0],gb[1],gb[2],Nx,Ny,Nz,newmark))
    
    # get the list of edges
    edgelist = [list(x) for x in set(tuple(x) for x in edgelist)]
    
    # output edges
    filename = 'edge_%d.txt' % dataid
    with open(os.path.join(path,filename),'w') as ofile:
        for edge in edgelist:
            ofile.write("%8d%8d\n" % (edge[0],edge[1])) 
