import numpy as np
import os
import shutil
import random

path = '/srv/home/daimy14/datageneration/genvoronoi/1'

# number of data in each group
number = 100
# group number
group  = 1
# get the conductivity of sigma_11 and sigma_22 of grain
gcond1 = 1.335e-5; gcond2 = 1.335e-5
# get the lower and upper limit of sigma_33
lowerlimit = 0.8; upperlimit = 1.2
# get the lower limit and upper limit of sigma of grain boundary
gblowerlimit = -3; gbupperlimit = 2
# get the list of conductivity
gcond3list = np.zeros((number,1))
gbcondlist = np.zeros((number,3))
# set the random number seed
random.seed(10)

for i in range(number):
    # specify folder name
    itotal = i + (group - 1) * number
    foldername = 'data_%d' % itotal
    os.makedirs(foldername, exist_ok = True)
    # copy neccessary files
    shutil.copy2('eulerAng_%d.in' % i, '%s/eulerAng.in' % foldername)
    shutil.copy2('struct_%d.in' % i, '%s/struct.in' % foldername)
    shutil.copy2('EffPropertyPoly.exe',foldername)
    shutil.copy2('parameterFormatted.in', foldername)
    # get random conductivity
    gcond3list[i, 0] = gcond1 * random.uniform(lowerlimit, upperlimit)
    gbcondlist[i, 0] = gcond1 * np.float_power(10, random.uniform(gblowerlimit, gbupperlimit))
    gbcondlist[i, 1] = gcond2 * np.float_power(10, random.uniform(gblowerlimit, gbupperlimit))
    gbcondlist[i, 2] = gcond3list[i] * np.float_power(10,random.uniform(gblowerlimit, gbupperlimit))
    # change parameter
    fin = open(os.path.join(foldername, 'parameterFormatted.in'),"rt")
    para = fin.read()
    para = para.replace('gcond3',str("{:.2E}".format(gcond3list[i,0]).replace("E-0","E-")))
    para = para.replace('gbcond1',str("{:.2E}".format(gbcondlist[i,0]).replace("E-0","E-")))
    para = para.replace('gbcond2',str("{:.2E}".format(gbcondlist[i,1]).replace("E-0","E-")))
    para = para.replace('gbcond3',str("{:.2E}".format(gbcondlist[i,2]).replace("E-0","E-")))
    fin.close()
    fin = open(os.path.join(foldername,'parameterFormatted.in'),"wt")
    fin.write(para)
    fin.close()
    os.chdir(path)

finalcond = np.concatenate((gcond3list, gbcondlist),axis=1)
np.savetxt("conductivity.txt",finalcond,fmt='%.2E')
