# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 23:09:27 2022

@author: daimi
"""

# This code saves data into npz file
import numpy as np
import os
import argparse

def loaddata(group, numdata, Nx, Ny, Nz, upperlimit, lowerlimit):
    # read conductivity file first
    foldername = '%d' % group
    condfile = os.path.join(foldername, 'finalconductivity.txt')
    gcond, gbcond, calcond = readcond(condfile, numdata, upperlimit, lowerlimit)
    # go through data points
    for n in range(numdata):
        # read structure file
        fstruct = '%s/struct_%d.in' % (foldername, n)
        struct = readstruct(fstruct, Nx, Ny, Nz)
        # read euler angle
        feulerang = '%s/eulerAng_%d.in' % (foldername, n)
        eulerang = readeulerang(feulerang, Nx, Ny, Nz)
        # get the conductivity matrix according to the structure id
        conductivity = getcond(struct,gcond[n,:],gbcond[n,:])
        # combine the conductivity matrix and the euler angle matrix
        gimage = getimage(conductivity, eulerang)
        if n == 0:
            gimagelist, targetlist = [gimage],[calcond[n,:]]
        else:
            gimagelist, targetlist = np.concatenate((gimagelist, [gimage])), \
                    np.concatenate((targetlist, [calcond[n,:]]))
        print('finish read data %d from group %d' % (n, group), flush = True)

    # change data to numpy array
    gimagelist = np.array(gimagelist)
    targetlist = np.array(targetlist)

    return gimagelist, targetlist

def readcond(condfile,numdata, upperlimit, lowerlimit):
    # conductivity from file
    cond = normcond(np.loadtxt(condfile),upperlimit, lowerlimit)
    gcond = np.copy(cond[:numdata,0:3])
    gbcond = npi.copy(cond[:numdata,3:6])
    calcond = np.copy(cond[:numdata,6:9])
    return gcond, gbcond, calcond

def readstruct(fstruct, Nx, Ny, Nz):
    # read structure data from file
    data = np.loadtxt(fstruct, skiprows = 1, dtype = np.int64)
    # get the structure data 
    struct = np.zeros((Nx, Ny, Nz))
    for i in range(len(data)):
        struct[data[i,0]-1,data[i,1]-1,data[i,2]-1] = data[i,3]
    
    return struct

def readeulerang(feulerang, Nx, Ny, Nz):
    # read euler ang data from file
    data = np.loadtxt(feulerang, skiprows = 1)
    # get the euler angle data
    eulerang = np.zeros((Nx, Ny, Nz,3))
    for i in range(len(data)):
        eulerang[int(data[i,0]-1), int(data[i,1]-1), int(data[i,2]-1),0:3] = data[i,3]/360, data[i,4]/180, data[i,5]/360
    
    return eulerang

def getcond(struct,gcond,gbcond):
    Nx, Ny, Nz = struct.shape
    cond = np.zeros((Nx, Ny, Nz, 3))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                if struct[i,j,k] == 1:
                    cond[i,j,k,:] = gcond
                else:
                    cond[i,j,k,:] = gbcond
    
    return cond

def getimage(cond, eulerang):
    gimage = np.concatenate((cond, eulerang), axis = 3)
    return gimage


def normcond(cond, upperlimit, lowerlimit):
    cond = np.log(cond)
    cond = (cond - lowerlimit) / (upperlimit - lowerlimit)

    return cond

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type = int, default = 1)
    parser.add_argument('--numdata', type = int, default = 100)
    parser.add_argument('--upperlimit', type = int, default = -2)
    parser.add_argument('--lowerlimit',type = int, default = -8)
    given_args = parser.parse_args()
    group = given_args.group
    numdata = given_args.numdata

    Nx = 64; Ny = 64; Nz = 64
    gimagelist, targetlist = loaddata(group, numdata, Nx, Ny, Nz, upperlimit, lowerlimit)
    ofilename = 'CNNdata_%d.npz' % group
    np.savez_compressed(ofilename, image = gimagelist, target = targetlist)

    CNNdata = np.load(ofilename)
    imagelist = CNNdata['image']
    targetlist = CNNdata['target']
    print("image size: ",imagelist.shape)
    print("target size: ",targetlist.shape)


