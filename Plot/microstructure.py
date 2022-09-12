# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 22:42:45 2022

@author: daimi
"""

import numpy as np
import matplotlib.pyplot as plt
from pyevtk.hl import gridToVTK


# convert euler ang to orientation
def convertcolor(euler1, euler2, euler3):
    # do rotation
    Rmatrix = np.zeros((3,3))
    Rmatrix[0,0] = np.cos(euler1) * np.cos(euler3) - np.cos(euler2) * np.sin(euler1) * np.sin(euler3)
    Rmatrix[0,1] = -np.cos(euler1) * np.sin(euler3) - np.cos(euler2) * np.cos(euler3) * np.sin(euler1)
    Rmatrix[0,2] = np.sin(euler1) * np.sin(euler2)
    Rmatrix[1,0] = np.cos(euler3) * np.sin(euler1) + np.cos(euler1) * np.cos(euler2) * np.sin(euler3)
    Rmatrix[1,1] = np.cos(euler1) * np.cos(euler2) * np.cos(euler3) - np.sin(euler1) * np.sin(euler3)
    Rmatrix[1,2] = -np.cos(euler1) * np.sin(euler2)
    Rmatrix[2,0] = np.sin(euler2) * np.sin(euler3)
    Rmatrix[2,1] = np.cos(euler3) * np.sin(euler2)
    Rmatrix[2,2] = np.cos(euler2)
    # calculate color
    colormax = max(abs(Rmatrix[0,2]), abs(Rmatrix[1,2]), abs(Rmatrix[2,2]))
    color1 = abs(Rmatrix[0,2])/colormax
    color2 = abs(Rmatrix[1,2])/colormax
    color3 = abs(Rmatrix[2,2])/colormax
    
    return color1, color2, color3



filename = 'eulerAng_20.in'
eulerin = np.loadtxt(filename, skiprows = 1)
filename = 'struct_20.in'
structin = np.loadtxt(filename, skiprows = 1)


Nx = 64; Ny = 64; Nz = 64
euler = np.zeros((Nx, Ny, Nz, 3))
struct = np.zeros((Nx, Ny, Nz))
graincolors = np.zeros((Nx, Ny, Nz, 3))


for n in range(eulerin.shape[0]):
    i, j, k = int(eulerin[n, 0])-1, int(eulerin[n, 1])-1, int(eulerin[n, 2])-1
    euler[i, j, k,:] = eulerin[n, 3:6] / 180 * np.pi
    struct[i, j, k] = structin[n, 3]
    if struct[i, j, k] == 1:
        graincolors[i, j, k, :] = convertcolor(euler[i, j, k, 0], euler[i, j, k, 1], euler[i, j, k, 2])
    else:
        graincolors[i, j, k, :] = [0, 0, 0]

x = np.zeros((Nx, Ny, Nz))
y = np.zeros((Nx, Ny, Nz))
z = np.zeros((Nx, Ny, Nz))
r = np.zeros((Nx, Ny, Nz))
g = np.zeros((Nx, Ny, Nz))
b = np.zeros((Nx, Ny, Nz))

for k in range(Nz):
    for j in range(Ny):
        for i in range(Nx):
            x[i, j, k] = i
            y[i, j, k] = j
            z[i, j, k] = k
            r[i, j, k] = graincolors[i, j, k, 0]
            g[i, j, k] = graincolors[i, j, k, 1]
            b[i, j, k] = graincolors[i, j, k, 2]
            
gridToVTK("./grain_20",x,y,z, pointData = {"RGBpoint": (r, g, b)})
            
    
    

 