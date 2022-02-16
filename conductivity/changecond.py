import numpy as np
import os
import shutil

condPowerList = np.arange(-8,-5,1)
path = '/ocean/projects/dmr180039p/mdai26/ML_polygrain_Grainboundary/realdatageneration'

number = 75

for i in range(number):
    for condPower in condPowerList:
        foldername = 'data_%d_cPowerminus_%d' % (i,abs(condPower))
        os.makedirs(foldername, exist_ok = True)
        shutil.copy2('Euler/eulerAng_%d.in' % i, '%s/eulerAng.in' % foldername)
        shutil.copy2('Structure/struct_%d.in' % i, '%s/struct.in' % foldername)
        shutil.copy2('EffPropertyPoly.exe',foldername)
        shutil.copy2('run.sh',foldername)
        shutil.copy2('parameterFormatted.in', foldername)
        fin = open(os.path.join(foldername, 'parameterFormatted.in'),"rt")
        para = fin.read()
        para = para.replace('CondPower',str(condPower))
        fin.close()
        fin = open(os.path.join(foldername,'parameterFormatted.in'),"wt")
        fin.write(para)
        fin.close()
        os.chdir(os.path.join(path,foldername))
        os.system("sbatch run.sh")
        os.chdir(path)
