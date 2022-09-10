import argparse
import os
import shutil
import sys
import time
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

from data import polygrainzipdata
from data import get_train_val_test_loader
from model import CrystalGraphConvNet

parser = argparse.ArgumentParser(description='Graph Neural Network for Polygrain conductivity prediction')
# parameters for loading data
parser.add_argument('--group', default=50, type=int, help='groups of data')
parser.add_argument('--max_node', default=400, type=int, help='maximum number of nodes')
# parameters for splitting data
parser.add_argument('--batch-size', default=10, type=int, help='mini-batch size (default: 256)')
parser.add_argument('--random_seed', default=5, type=int, help='random seed for splitting data')
parser.add_argument('--train_ratio', default=0.8, type=float, help='ratio for training data points')
parser.add_argument('--val_ratio', default=0.1, type=float, help='ratio of validation data points')
# parameters for model
parser.add_argument('--node-fea-len', default=64, type=int, help='number of hidden node features in conv layers')
parser.add_argument('--h-fea-len', default=128, type=int, help='number of hidden features after pooling')
parser.add_argument('--n-conv', default=2, type=int, help='number of conv layers')
parser.add_argument('--n-h', default=2, type=int, help='number of hidden layers after pooling')
# parameters for using CUDA or not
parser.add_argument('--disable-cuda', action='store_true',help='Disable CUDA')
# parameters for optimizer
parser.add_argument('--optim', default='Adam', type=str, help='choose an optimizer, SGD or Adam, (default: SGD)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,help='weight decay (default: 0)')
parser.add_argument('--lr', '--learning-rate', default=1e-6, type=float,help='initial learning rate (default: 0.01)')
# parameters for checking points
parser.add_argument('--resume', default='checkpoints/', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
# parameters for training
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')



parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: '
                                      '[100])')
# get the arguments
args = parser.parse_args(sys.argv[1:])

# check if cuda is available
args.cuda = not args.disable_cuda and torch.cuda.is_available()

# we use regression here
best_mae_error = 1e10


def main():
    global args, best_mae_error
    
    # keep the same results on different devices
    os.environ['PYTHONHASHargs.seed'] = str(args.random_seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # load data
    train_ds = polygrainzipdata('../../datasplit/GNNtraindata_unscaled.npz')
    valid_ds = polygrainzipdata('../../datasplit/GNNvaliddata_unscaled.npz')
    test_ds = polygrainzipdata('../../datasplit/GNNtestdata_unscaled.npz')
    # data split
    train_loader, val_loader, test_loader = get_train_val_test_loader(train_ds, valid_ds, test_ds, args.batch_size, args.cuda)

    # build model
    # get the number of node feature and edge feature
    nfeature, _, efeature, _ = train_ds[0]
    orig_atom_fea_len = nfeature.shape[1]
    edge_fea_len = efeature.shape[2]
    # build model
    model = CrystalGraphConvNet(orig_atom_fea_len, edge_fea_len, args.max_node,
                                node_fea_len=args.node_fea_len,
                                n_conv=args.n_conv,
                                h_fea_len=args.h_fea_len,
                                n_h=args.n_h)
    
    model = model.float()
    # not sure if this is suitable, just keep it for now.
    if args.cuda:
        model.cuda()

    # define loss func and optimizer
    # change to MAE
    criterion = nn.L1Loss()
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.lr,
                               weight_decay=args.weight_decay)
    else:
        raise NameError('Only SGD or Adam is allowed as --optim')
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume),flush=True)
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mae_error = checkpoint['best_mae_error']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']), flush = True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume), flush = True)

    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones,gamma=0.1)

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        mae_error = validate(val_loader, model, criterion, epoch)

        if mae_error != mae_error:
            print('Exit due to NaN', flush = True)
            sys.exit(1)

        scheduler.step()

        # remember the best mae_eror and save checkpoin
        is_best = mae_error < best_mae_error
        best_mae_error = min(mae_error, best_mae_error)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            'args': vars(args)
        }, is_best)

    # test best model
    print('---------Evaluate Model on Test Set---------------', flush = True)
    best_checkpoint = torch.load('model_best.pth.tar')
    model.load_state_dict(best_checkpoint['state_dict'])
    validate(test_loader, model, criterion, epoch, test=True)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (nfeature, neighlist, efeature, targetlist) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            nfeature = Variable(nfeature.cuda(non_blocking=True).float())
            neighlist = Variable(neighlist.cuda(non_blocking=True).float())
            efeature = Variable(efeature.cuda(non_blocking=True).float())
            targetlist = Variable(targetlist.cuda(non_blocking=True).float())
        else:
            nfeature = Variable(nfeature.float())
            neighlist = Variable(neighlist.float())
            efeature = Variable(efeature.float())
            targetlist = Variable(targetlist.float())
        if args.cuda:
            targetlist = Variable(targetlist.cuda(non_blocking=True).float())
        else:
            targetlist = Variable(targetlist.float())

        # compute output
        output = model(nfeature, neighlist, efeature)
        loss = criterion(output, targetlist)

        # measure accuracy and record loss
        mae_error = mae(output.data.cpu(), targetlist.cpu())
        losses.update(loss.data.cpu(), targetlist.size(0))
        mae_errors.update(mae_error, targetlist.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch: {epoch:6d}\t'
          'Time {batch_time.avg:.3f} ({batch_time.count:6d})\t'
          'Data {data_time.avg:.3f} ({data_time.count:6d})\t'
          'Loss {loss.avg:.4f} ({loss.count:6d})\t'
          'MAE {mae_errors.avg:.3e} ({mae_errors.count:6d})'.format(
           epoch=epoch, batch_time=batch_time,
           data_time=data_time, loss=losses, mae_errors=mae_errors), flush = True)


def validate(val_loader, model, criterion, epoch, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()
    if test:
        test_targets = []
        test_preds = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (nfeature, neighlist, efeature, targetlist) in enumerate(val_loader):
        if args.cuda:
            with torch.no_grad():
                nfeature = Variable(nfeature.cuda(non_blocking=True).float())
                neighlist = Variable(neighlist.cuda(non_blocking=True).float())
                efeature = Variable(efeature.cuda(non_blocking=True).float())
                targetlist = Variable(targetlist.cuda(non_blocking=True).float())
        else:
            with torch.no_grad():
                nfeature = Variable(nfeature.float())
                neighlist = Variable(neighlist.float())
                efeature = Variable(efeature.float())
                targetlist = Variable(targetlist.float())
        if args.cuda:
            with torch.no_grad():
                targetlist = Variable(targetlist.cuda(non_blocking=True).float())
        else:
            with torch.no_grad():
                targetlist = Variable(targetlist.float())

        # compute output
        output = model(nfeature, neighlist, efeature)
        loss = criterion(output, targetlist)

        # measure accuracy and record loss
        mae_error = mae(output.data.cpu(), targetlist.cpu())
        losses.update(loss.data.cpu().item(), targetlist.size(0))
        mae_errors.update(mae_error, targetlist.size(0))
        if test:
            test_pred = output.data.cpu()
            test_target = targetlist
            #test_pred = denormalize(test_pred)
            #test_target = denormalize(targetlist)
            test_preds += test_pred.tolist()
            test_targets += test_target.tolist()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch: {epoch:6d}\t'
          'Time {batch_time.avg:.3f} ({batch_time.count:6d})\t'
          'Loss {loss.avg:.4f} ({loss.count:6d})\t'
          'MAE {mae_errors.avg:.3e} ({mae_errors.count:6d})'.format(
           epoch=epoch, batch_time=batch_time, loss=losses,
           mae_errors=mae_errors), flush = True)

    if test:
        star_label = '**'
        with open('test_results.csv', 'w') as f:
            writer = csv.writer(f)
            for target, pred in zip(test_targets,test_preds):
                writer.writerow((target, pred))
    else:
        star_label = '*'
        
    #print(' {star} MAE {mae_errors.avg:.3e}'.format(star=star_label,
    #                                                    mae_errors=mae_errors), flush = True)
    return mae_errors.avg


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    # do de-normalization
    #prediction = denormalize(prediction)
    #target = denormalize(target)

    return torch.mean(torch.abs(target - prediction))

def denormalize(target):
    norm = np.load('norm.npz', allow_pickle = True)['norm']
    t_norm = torch.from_numpy(norm)
    target = target * t_norm[1] + t_norm[0]
    return target

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')



if __name__ == '__main__':
    main()
