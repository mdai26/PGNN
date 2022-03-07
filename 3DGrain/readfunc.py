# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 15:44:43 2022

@author: daimi
"""
import numpy as np
import os

def readmarkfile(path, infilename1, Nx, Ny, Nz):
    # open file and split lines
    infile = open(os.path.join(path,infilename1),'r')
    lines = infile.readlines()[1:]
    lines = [line.split() for line in lines]
    
    # read mark from the file
    mark = np.zeros((Nx,Ny,Nz))
    for i in range(Nx*Ny*Nz):
        mark[int(lines[i][0])-1, int(lines[i][1])-1, int(lines[i][2])-1] = int(lines[i][3])
    
    # change grain to 1 and grain boundary to 0
    mark_new = np.zeros((Nx, Ny, Nz))    
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                if mark[i, j, k] == 0:
                    mark_new[i, j, k] = 0
                else:
                    mark_new[i, j, k] = 1
        
    return mark_new

def readeulerfile(path, infilename2, Nx, Ny, Nz):
    # open file and split lines
    infile = open(os.path.join(path,infilename2),'r')
    lines = infile.readlines()[1:]
    lines = [line.split() for line in lines]
    
    # read mark from the file
    eulerang = np.zeros((Nx,Ny,Nz,3))
    for i in range(Nx*Ny*Nz):
        eulerang[int(lines[i][0])-1, int(lines[i][1])-1, int(lines[i][2])-1,0:3] = lines[i][3:6]   
        
    return eulerang
