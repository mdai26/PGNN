import numpy as np
import os
import shutil
import random

path = '/srv/home/daimy14/datageneration/genvoronoi/calelastic'

# number of data in each group
number = 250
# get the elastic parameters of grain
gc11 = 1.87e11; gc12 = 7.51e10; gc44 = 7.10e10
gblowerlimit = -3; gbupperlimit = 0
# get the list of conductivity
gbelalist = np.zeros((number,3))
# set the random number seed
random.seed(10)

fsequence = '../datasplit/trainid.txt'
sequence = np.array([int(s) for s in np.loadtxt(fsequence)])

for i in range(number):
    dataid = sequence[i]
    group = int(dataid/100) + 1
    # data folder name
    groupfolder = '../%d/data_%d' % (group, dataid)
    # specify folder name
    foldername = 'data_%d' % i
    os.makedirs(foldername, exist_ok = True)
    # copy neccessary files
    shutil.copy2('%s/eulerAng.in' % groupfolder, '%s/eulerAng.in' % foldername)
    shutil.copy2('%s/struct.in' % groupfolder, '%s/struct.in' % foldername)
    shutil.copy2('EffPropertyPoly.exe',foldername)
    shutil.copy2('parameterFormatted.in', foldername)
    # get random conductivity
    gbelalist[i, 0] = gc11 * np.float_power(10, random.uniform(gblowerlimit, gbupperlimit))
    gbelalist[i, 1] = gc12 * np.float_power(10, random.uniform(gblowerlimit, gbupperlimit))
    gbelalist[i, 2] = gc44 * np.float_power(10,random.uniform(gblowerlimit, gbupperlimit))
    # change parameter
    fin = open(os.path.join(foldername, 'parameterFormatted.in'),"rt")
    para = fin.read()
    para = para.replace('gbc11',str("{:.2E}".format(gbelalist[i,0])).replace("E+0","E").replace("E+","E"))
    para = para.replace('gbc12',str("{:.2E}".format(gbelalist[i,1])).replace("E+0","E").replace("E+","E"))
    para = para.replace('gbc44',str("{:.2E}".format(gbelalist[i,2])).replace("E+0","E").replace("E+","E"))
    fin.close()
    fin = open(os.path.join(foldername,'parameterFormatted.in'),"wt")
    fin.write(para)
    fin.close()
    os.chdir(path)

finalela = np.concatenate((sequence[:number].reshape((-1,1)), gbelalist),axis=1)
np.savetxt("elastic.txt",finalela,fmt='%.2E')
