import pandas as pd
import numpy as np
import os
import re

def readcond(filename):
    infile = open(filename,'r')
    lines = infile.readlines()
    match_number = re.compile('\s-?[0-9]+[0-9]*.?[0-9]*E-?\+?[0-9]+\s')
    #print(lines[0].split(" "))
    line1 = [float(x) for x in re.findall(match_number, lines[0])]
    line2 = [float(x) for x in re.findall(match_number, lines[1])]
    line3 = [float(x) for x in re.findall(match_number, lines[2])]
    #print(line1,line2,line3)
    return np.array([line1[0], line2[1], line3[2]])

# path
path = '/srv/home/daimy14/datageneration/genvoronoi/1'
# number of data in each group
number = 100
# group number
group = 1
# specify the conductivity array
cond = np.zeros((number, 9))
# get sigma11 and sigma22 for grain
cond[:,0] = 1.335e-5
cond[:,1] = 1.335e-5
# get signma33 for each grain, sigma for grain boundary
cond[:,2:6] = np.loadtxt("conductivity.txt")
# get calculated conductivity results from each folder
for i in range(number):
    dataid = i + (group - 1) * number
    foldername = 'data_%d' % dataid
    filename = os.path.join(foldername, 'effElectricalConductivity.dat')
    if os.path.exists(filename):
        cond[i,6], cond[i,7], cond[i,8] = readcond(filename)
    else:
        print("%s not exist" % filename)

np.savetxt("finalconductivity.txt",cond,fmt='%.3E')
