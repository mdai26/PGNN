import numpy as np
import random

def loadGNNdata(group):
    for g in range(1, group+1):
        filename = '../GNNdata_%d.npz' % g
        GNNdata = np.load(filename)
        print("load GNN data from group %d" % g, flush = True)
        if g == 1:
            nfeature = GNNdata['nfeature']
            neighblist = GNNdata['neighblist']
            efeature = GNNdata['efeature']
            targetlist = GNNdata['targetlist']
        else:
            nfeature = np.append(nfeature, GNNdata['nfeature'],axis=0)
            neighblist = np.append(neighblist, GNNdata['neighblist'], axis=0)
            efeature = np.append(efeature, GNNdata['efeature'], axis=0)
            targetlist = np.append(targetlist, GNNdata['targetlist'], axis=0)
    return nfeature, neighblist, efeature, targetlist

def loadCNNdata(group):
    for i in range(1, group + 1):
        filename = '../CNNdata_%d.npz' % i
        print("load CNN data from group %d" % i, flush = True)
        CNNdata = np.load(filename)
        if i == 1:
            imagelist = CNNdata['image']
            targetlist = CNNdata['target']
        else:
            imagelist = np.append(imagelist, CNNdata['image'], axis = 0)
            targetlist = np.append(targetlist, CNNdata['target'], axis = 0)
    return imagelist, targetlist


def getsequence(number):
    dataid = np.arange(number)
    random.shuffle(dataid)
    trainid = dataid[:int(0.8*number)]
    validid = dataid[int(0.8*number):int(0.9*number)]
    testid = dataid[int(0.9*number):]

    return trainid, validid, testid

# main
group = 50
# get GNN data
nfeature, neighblist, efeature, targetlist = loadGNNdata(group)
# get sequence
random_seed = 25
np.random.seed(random_seed)
random.seed(random_seed)
trainid, validid, testid = getsequence(targetlist.shape[0])
# save GNN data
np.savez_compressed('GNNtraindata.npz', nfeature = nfeature[trainid], neighblist = neighblist[trainid], efeature = efeature[trainid], targetlist = targetlist[trainid])
np.savez_compressed('GNNvaliddata.npz', nfeature = nfeature[validid], neighblist = neighblist[validid], efeature = efeature[validid], targetlist = targetlist[validid])
np.savez_compressed('GNNtestdata.npz', nfeature = nfeature[testid], neighblist = neighblist[testid], efeature = efeature[testid], targetlist = targetlist[testid])
# save other file
np.savetxt('trainid.txt', trainid)
np.savetxt('validid.txt', validid)
np.savetxt('testid.txt', testid)
np.savetxt('alltarget.txt', targetlist)
# get CNN data
imagelist, targetlist = loadCNNdata(group)
# save CNN data
np.savez_compressed('CNNtraindata.npz', imagelist = imagelist[trainid], targetlist = targetlist[trainid])
np.savez_compressed('CNNvaliddata.npz', imagelist = imagelist[validid], targetlist = targetlist[validid])
np.savez_compressed('CNNtestdata.npz', imagelist = imagelist[testid], targetlist = targetlist[testid])
