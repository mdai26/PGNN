# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:49:44 2023

@author: daimi
"""

import numpy as np

'''
get eulerangles and featureid from EBSD data
Convert the whole 3D microstructures into 
several RVE with the size of 64 x 64 x 64.
'''
def getEBSDdata(eulerfile, featurefile):
    ix, iy, iz = 549, 420, 526
    # read euler data
    euler = np.zeros((ix, iy, iz,3))
    eulerdata = np.loadtxt(eulerfile, delimiter=',')
    # read feature id
    featureid = np.zeros((ix, iy, iz),dtype = int)
    featuredata = np.loadtxt(featurefile)
    # get the results in correct form
    for z in range(iz):
        for y in range(iy):
            for x in range(ix):
                euler[x, y, z, :] = eulerdata[z*iy*ix + y*ix + x,:]
                featureid[x, y, z] = int(featuredata[z*iy*ix + y*ix + x])
    # get the microlist
    # divide the whole microstructure into several 3d cube
    # through link: https://stackoverflow.com/questions/39429900/split-a-3d-numpy-array-into-3d-blocks
    d = 70
    featureid = featureid[:420,:420,:420]
    microlist = featureid.reshape(-1,d,6,d,6,d).transpose(0,2,4,1,3,5).reshape(-1,d,d,d)
    # get the eulerlist
    euler = euler[:420,:420,:420,:]
    eulerlist = euler.reshape(-1,d,6,d,6,d,3).transpose(0,2,4,1,3,5,6).reshape(-1,d,d,d,3)
       
    return microlist, eulerlist

eulerfile = 'EulerAngles.txt'
featurefile = 'FeatureIds.txt'
microlist, eulerlist = getEBSDdata(eulerfile, featurefile)
np.save('microlist', microlist)
np.save('eulerlist', eulerlist)

