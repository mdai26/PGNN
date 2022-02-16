import pandas as pd
import numpy as np
import os
import re

def readcond(filename):
    infile = open(filename,'r')
    lines = infile.readlines()
    match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
    line1 = [float(x) for x in re.findall(match_number, lines[0])]
    line2 = [float(x) for x in re.findall(match_number, lines[1])]
    line3 = [float(x) for x in re.findall(match_number, lines[2])]
    #print(line1[0],line2[1],line3[2])
    return np.array([line1[0], line2[1], line3[2]])


condPowerList = np.arange(-8,-5,1)
path = '/ocean/projects/dmr180039p/mdai26/ML_polygrain_Grainboundary/realdatageneration'

number = 75

df = pd.DataFrame(columns = ['id','cond1','1sigma11','1sigma22','1sigma33','cond2','2sigma11','2sigma22','2sigma33','cond3','3sigma11','3sigma22','3sigma33'])

for i in range(number):
    sigma11, sigma22, sigma33 = np.zeros(3), np.zeros(3), np.zeros(3)
    for j in range(3):
        foldername = 'data_%d_cPowerminus_%d' % (i, abs(condPowerList[j]))
        filename = os.path.join(foldername, 'effElectricalConductivity.dat')
        if os.path.exists(filename):
            sigma11[j], sigma22[j], sigma33[j] = readcond(filename)
    df = df.append({'id':i, 'cond1': condPowerList[0], '1sigma11': sigma11[0], '1sigma22': sigma22[0], '1sigma33': sigma33[0], 'cond2': condPowerList[1], '2sigma11': sigma11[1], '2sigma22': sigma22[1], '2sigma33': sigma33[1], 'cond3': condPowerList[2], '3sigma11': sigma11[2], '3sigma22': sigma22[2], '3sigma33': sigma33[2]},ignore_index=True)

dfilename = 'conductivity.csv'
df.to_csv(os.path.join(path, dfilename))
