# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 17:22:59 2022

@author: daimi
"""

import os

def outputeuler(eulerang,path,dataid,Nx,Ny,Nz):
    filename = 'eulerAng_%d.in' % dataid
    with open(os.path.join(path,filename),'w') as ofile:
        ofile.write("%6d%6d%6d\n" % (Nx, Ny, Nz))
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    ofile.write("%6d%6d%6d%8.2f%8.2f%8.2f\n" % (i+1,j+1,k+1,eulerang[i,j,k,0],eulerang[i,j,k,1],eulerang[i,j,k,2]))
    
    
def outputstruct(mark,path,dataid,Nx,Ny,Nz):
    filename = 'struct_%d.in' % dataid
    with open(os.path.join(path,filename),'w') as ofile:
        ofile.write("%6d%6d%6d\n" % (Nx, Ny, Nz))
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    if mark[i, j, k] == 0:
                        ofile.write("%6d%6d%6d%6d\n" % (i+1,j+1,k+1,2))
                    else:
                        ofile.write("%6d%6d%6d%6d\n" % (i+1,j+1,k+1,1))