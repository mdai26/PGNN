import numpy as np
import os
import shutil

klist = np.arange(0.2,1.2,0.2)
ninitgorientlist = np.arange(12,60,12)
path = '/srv/home/daimy14/graingrowth/parameter_test'

for k in klist:
    for norient in ninitgorientlist:
        foldername = 'k_%.2f_norient_%d' % (k, norient)
        os.makedirs(foldername, exist_ok = True)
        shutil.copy2('3dGG_OpenMP',foldername)
        shutil.copy2('GG.sh',foldername)
        shutil.copy2('read.in',foldername)
        fin = open(os.path.join(foldername,'read.in'),"rt")
        para = fin.read()
        para = para.replace('kxx',str(k))
        para = para.replace('kyy',str(k))
        para = para.replace('kzz',str(k))
        para = para.replace('ninitgorient',str(norient))
        fin.close()
        fin = open(os.path.join(foldername,'read.in'),"wt")
        fin.write(para)
        fin.close()
        os.chdir(os.path.join(path,foldername))
        os.system("sbatch GG.sh")
        os.chdir(path)



