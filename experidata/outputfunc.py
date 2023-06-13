# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 19:46:26 2022

@author: daimi
"""
import numpy as np
from pyevtk.hl import gridToVTK


def outputeuler(eulerang, dataid, Nx, Ny, Nz):
    filename = 'eulerAng_%d.in' % dataid
    with open(filename,'w') as ofile:
        ofile.write("%6d%6d%6d\n" % (Nx, Ny, Nz))
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    ofile.write("%6d%6d%6d%8.2f%8.2f%8.2f\n" % (i+1,j+1,k+1,eulerang[i,j,k,0],eulerang[i,j,k,1],eulerang[i,j,k,2]))
    
    
def outputstruct(mark, flag, dataid, Nx, Ny, Nz):
    if flag == "struct":
        filename = 'struct_%d.in' % dataid
    elif flag == "grainmark":
        filename = 'grainmark_%d.in' % dataid
    with open(filename,'w') as ofile:
        ofile.write("%6d%6d%6d\n" % (Nx, Ny, Nz))
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    ofile.write("%6d%6d%6d%6d\n" % (i+1,j+1,k+1, mark[i,j,k]))



def outputneighbor(neighbor, dataid, Nx, Ny, Nz):
    filename = 'neighbor_%d.txt' % dataid
    np.savetxt(filename, neighbor, fmt='%d')
    
                        
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