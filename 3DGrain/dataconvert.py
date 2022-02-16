# 1/12/2022
# written by Minyi
# This is main code for data conversion

import numpy as np
import pandas as pd
import os
from readfunc import readmarkfile, readeulerfile
from outputfunc import outputeuler, outputstruct
from ngrain import ngraincal
from featurecal import extractfeature

# specify system dimension
Nx = 64; Ny = 64; Nz = 64

# specify the list of the three parameters
klist = np.arange(0.2,1.2,0.2)
niorientlist = np.arange(12,60,12)
nsteplist = np.arange(2000,10000,2000)

# create an empty DataFrame
df = pd.DataFrame(columns = ['kxx','kyy','kzz','norient','step','total grains'])

# set path and creteria for the data
path = '/srv/home/daimy14/graingrowth/parameter_test'
ngrainmin = 10
ngrainmax = 300

# specify intial id
dataid = 0
# go through the loops
for k in klist:
    for niorient in niorientlist:
        for nstep in nsteplist:
            # specify file name
            infilename1 = 'k_%.2f_norient_%d/grainmark_%08d.txt' % (k, niorient, nstep)
            if os.path.exists(os.path.join(path,infilename1)) == False:
                print("%s not exist" % infilename1)
                break
            # read mark from file
            mark = readmarkfile(path, infilename1, Nx, Ny, Nz)
            mark = mark.astype(int)
            print("read mark complete!\n")
            # calculate number of files
            ngrain,newmark = ngraincal(mark,Nx,Ny,Nz)
            print("number of grain calculation complete!\n")
            # check if the data is suitable
            #if (ngrain >= ngrainmin) and (ngrain <= ngrainmax):
            # add information to the data framework
            df = df.append({'kxx' : k, 'kyy': k, 'kzz' : k, \
                            'norient' : niorient, \
                            'step' : nstep, \
                            'total grains' : ngrain},ignore_index=True)
            # read euler angles from file
            infilename2 = 'k_%.2f_norient_%d/eulerAng_%08d.in' % (k, niorient, nstep)
            eulerang = readeulerfile(path, infilename2, Nx, Ny, Nz)
            print("read euler angles complete!\n")
            # output euler angle and struct id
            outputeuler(eulerang, path, dataid, Nx, Ny, Nz)
            outputstruct(mark, path, dataid, Nx, Ny, Nz)
            print("output euler and struct complete!\n")
            # calculate grain and grain boundary feature
            extractfeature(newmark, eulerang, ngrain, path, dataid, Nx, Ny, Nz)
            print("extractfeature completer!\n")
            dataid = dataid + 1

dffilename = 'data.csv'
df.to_csv(os.path.join(path,dffilename))
