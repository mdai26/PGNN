# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 15:54:40 2022

@author: daimi
"""
import numpy as np

# find neighbors of points (i,j)
# implement periodic boundary condition
def findneigbhors(i,j,k,Nx,Ny,Nz):
    # x +
    ip = i + 1
    if ip >= Nx:
        ip = ip - Nx
    # x -
    im = i - 1
    if im <0 :
        im = im + Nx
    # y +
    jp = j + 1
    if jp >= Ny:
        jp = jp - Ny
    # y -
    jm = j - 1
    if jm < 0 :
        jm = jm + Ny
    # z +
    kp = k + 1
    if kp >= Nz:
        kp = kp - Nz
    # z -
    km = k - 1
    if km < 0:
        km = km + Nz
        
    return [[ip,j,k],[im,j,k],[i,jp,k],[i,jm,k],[i,j,kp],[i,j,km]]


def ngraincal(mark,Nx,Ny,Nz):
    # get the list of grain bounary points
    l_test = []
    visited = np.zeros((Nx,Ny,Nz))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                if mark[i,j,k] == 1:
                    l_test.append([i,j,k])
                    visited[i,j,k] = 1
    
    # define the new mark matrix
    newmark = np.zeros((Nx,Ny,Nz))
    # this is the initial label
    mark0 = 1
    
    # repeat the labeling until all the points are marked
    while l_test:
        # choose the first element of the list as a starting point
        l_layer = [l_test[0]]
        # tree implenmentation: repeat until all neighbors are found for the initial point
        while l_layer:
            # add mark for layer of the tree
            # remove marked point from the original list
            for element in l_layer:
                newmark[element[0],element[1],element[2]] = mark0
                l_test.remove(element)
                visited[element[0],element[1],element[2]] = 0
            nl_layer = []
            for element in l_layer:
                posspoints = findneigbhors(element[0],element[1],element[2],Nx,Ny,Nz)
                # check if six neighbors is visited or not
                for point in posspoints:
                    if visited[point[0],point[1],point[2]] == 1:
                        visited[point[0],point[1],point[2]] =0
                        nl_layer.append(point)
            
            l_layer = nl_layer
        
        mark0 = mark0 + 1
    
    return mark0 - 1, newmark.astype(int) 
