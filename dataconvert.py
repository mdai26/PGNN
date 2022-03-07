# 1/12/2022
# written by Minyi
# This is main code for data conversion
# Second version: 3/6/2022
# update position calculation and finding edges

import numpy as np
import pandas as pd
import os
from readfunc import readmarkfile, readeulerfile
from outputfunc import outputeuler, outputstruct, outputvtk
from ngrain import ngraincal
from featurecal import extractfeature

# specify system dimension
Nx = 64; Ny = 64; Nz = 64

# specify the list of the three parameters
kxxlist = np.arange(0.2,1.0,0.2)
kyylist = np.arange(0.2,1.0,0.2)
kzzlist = np.arange(0.2,1.0,0.2)
niorientlist = np.arange(12,48,12)
nsteplist = np.arange(2000,10000,2000)

# create an empty DataFrame
df = pd.DataFrame(columns = ['kxx','kyy','kzz','norient','step','total grains'])

# set path and creteria for the data
path = '/srv/home/daimy14/datageneration/genmicro'
ngrainmin = 10
ngrainmax = 500

# specify intial id
dataid = 0
# go through the loops
for kxx in kxxlist:
    for kyy in kyylist:
        for kzz in kzzlist:
            for norient in niorientlist:
                for nstep in nsteplist:
                    # specify file name
                    infilename1 = 'kxx_%.2f_kyy_%.2f_kzz_%.2f_norient_%d/grainmark_%08d.txt' % (kxx, kyy, kzz, norient, nstep)
                    if os.path.exists(os.path.join(path,infilename1)) == False:
                        print("%s not exist" % infilename1, flush = True)
                        break
                    # read mark from file
                    mark = readmarkfile(path, infilename1, Nx, Ny, Nz)
                    mark = mark.astype(int)
                    # calculate number of files (ngraincal tested !)
                    ngrain,newmark = ngraincal(mark,Nx,Ny,Nz)
                    # check if the data is suitable
                    if (ngrain >= ngrainmin) and (ngrain <= ngrainmax):
                        print('dataid: %d, file name: kxx_%.2f_kyy_%.2f_kzz_%.2f_norient_%d_nstep_%d' % (dataid, kxx, kyy, kzz, norient, nstep), flush = True)
                        # add information to the data framework
                        df = df.append({'kxx' : kxx, 'kyy': kyy, 'kzz' : kzz, \
                                        'norient' : norient, \
                                        'step' : nstep, \
                                        'total grains' : ngrain}, ignore_index=True)
                        # read euler angles from file
                        infilename2 = 'kxx_%.2f_kyy_%.2f_kzz_%.2f_norient_%d/eulerAng_%08d.in' % (kxx, kyy, kzz, norient, nstep)
                        eulerang = readeulerfile(path, infilename2, Nx, Ny, Nz)
                        # output euler angle and struct id
                        outputeuler(eulerang, path, dataid, Nx, Ny, Nz)
                        outputstruct(mark, path, dataid, Nx, Ny, Nz)
                        # output vtk file of mark and newmark
                        outputvtk(mark, newmark, path, dataid, Nx, Ny, Nz)
                        # calculate grain and grain boundary feature
                        extractfeature(newmark, eulerang, ngrain, path, dataid, Nx, Ny, Nz)
                        dataid = dataid + 1
                    else:
                        print('exceed limit! file name: kxx_%.2f_kyy_%.2f_kzz_%.2f_norient_%d_nstep_%d' % (kxx, kyy, kzz, norient, nstep), flush = True)

dffilename = 'data.csv'
df.to_csv(os.path.join(path,dffilename))
