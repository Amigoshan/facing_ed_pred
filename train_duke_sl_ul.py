# Compare the sl using different number of labeled data
# This file is modified from train_semi_reg_joinloss.py
# Change to the mobilenet
# Supervised baseline

# July 2018: train on DukeMTMC

import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from os.path import join
from math import exp, sqrt
import random

from utils import loadPretrain2, loadPretrain, groupPlot
from facingDroneLabelData import FacingDroneLabelDataset
from facingDroneUnlabelData import FacingDroneUnlabelDataset
# from facingLabelData import FacingLabelDataset
from trackingLabelData import TrackingLabelDataset
# from trackingUnlabelData import TrackingUnlabelDataset
# from StateEncoderDecoder import EncoderReg_norm as EncoderReg
# from MobileReg import MobileReg
from StateEncoderDecoder import EncoderReg


exp_prefix = '52_1_'
preTrainModel = 'models_facing/1_2_encoder_decoder_facing_leaky_50000.pkl'
predictModel = 'models_facing/'+exp_prefix+'ed_reg_sl'
imgoutdir = 'resimg_facing'
datadir = 'data_facing'
datasetdir = '/datadrive/datasets'
Lr_label = 0.0005 
batch = 64
trainstep = 40000
showiter = 50
snapshot = 5000
train_layer_num = 0

unlabel_batch = 24 #32
lamb = 0.0
thresh = 0.01
train_layer_num = 0

hiddens = [3,16,32,32,64,64,128,256] 
kernels = [4,4,4,4,4,4,3]
paddings = [1,1,1,1,1,1,0]
strides = [2,2,2,2,2,2,1]

encoderReg = EncoderReg(hiddens, kernels, strides, paddings, actfunc='leaky')
# encode the input using pretrained model
print 'load pretrained...'
net=loadPretrain(encoderReg,preTrainModel)
net.cuda()


paramlist = list(net.parameters())
criterion = nn.MSELoss()
# kl_criterion = nn.KLDivLoss()
# regOptimizer = optim.SGD(predictNet.parameters(), lr = Lr, momentum=0.9)
regOptimizer = optim.Adam(paramlist[-train_layer_num:], lr = Lr_label)

valset1 = FacingDroneLabelDataset(imgdir=join(datasetdir,'droneData/val'))
valset2 = TrackingLabelDataset(filename='/datadrive/person/DukeMTMC/test_heading_gt.txt')
imgdataset = TrackingLabelDataset(filename='/datadrive/person/DukeMTMC/trainval_duke_10.txt', data_aug=True)
unlabelset = FacingDroneUnlabelDataset(imgdir='/datadrive/person/DukeMTMC/heading',batch = unlabel_batch, data_aug=True, include_all=True)
valnum = 50
dataloader = DataLoader(imgdataset, batch_size=batch, shuffle=True, num_workers=2)
valloader1 = DataLoader(valset1, batch_size=valnum, shuffle=True, num_workers=2)
valloader2 = DataLoader(valset2, batch_size=valnum, shuffle=True, num_workers=2)
unlabelloader = DataLoader(unlabelset, batch_size=1, shuffle=True, num_workers=2)

def train_label_unlabel(encoderReg, sample, unlabel_sample, regOptimizer, criterion, lamb):

    # labeled data
    inputImgs = sample['img']
    labels = sample['label']
    inputState = Variable(inputImgs,requires_grad=True)
    targetreg = Variable(labels,requires_grad=False)

    inputState = inputState.cuda()
    targetreg = targetreg.cuda()

    # forward + backward + optimize
    output, encode = encoderReg(inputState)
    loss_label = criterion(output, targetreg)

    # unlabeled data
    imgseq = unlabel_sample.squeeze()
    inputState_unlabel = Variable(imgseq,requires_grad=True).cuda()
    inputState_unlabel = inputState_unlabel.cuda()

    output_unlabel, x_encode = encoderReg(inputState_unlabel)

    loss_unlabel = Variable(torch.Tensor([0])).cuda()
    for ind1 in range(unlabel_batch-5): # try to make every sample contribute
        # randomly pick two other samples
        ind2 = random.randint(ind1+2, unlabel_batch-1) # big distance
        ind3 = random.randint(ind1+1, ind2-1) # small distance

        # target1 = Variable(x_encode[ind2,:].data, requires_grad=False).cuda()
        # target2 = Variable(x_encode[ind3,:].data, requires_grad=False).cuda()
        # diff_big = criterion(x_encode[ind1,:], target1) #(output_unlabel[ind1]-output_unlabel[ind2])*(output_unlabel[ind1]-output_unlabel[ind2])
        diff_big = (output_unlabel[ind1]-output_unlabel[ind2])*(output_unlabel[ind1]-output_unlabel[ind2])
        diff_big = diff_big.sum()/2.0
        # diff_small = criterion(x_encode[ind1,:], target2) #(output_unlabel[ind1]-output_unlabel[ind3])*(output_unlabel[ind1]-output_unlabel[ind3])
        diff_small = (output_unlabel[ind1]-output_unlabel[ind3])*(output_unlabel[ind1]-output_unlabel[ind3])
        diff_small = diff_small.sum()/2.0
        # import ipdb; ipdb.set_trace()
        loss_unlabel = loss_unlabel + (diff_small+thresh-diff_big).clamp(0)

    # loss = encode.sum()
    loss = loss_label + loss_unlabel * lamb #+ normloss * lamb2

    # zero the parameter gradients
    regOptimizer.zero_grad()
    # loss_label.backward()
    loss.backward()
    regOptimizer.step()

    return loss_label.data[0], loss_unlabel.data[0], loss.data[0]#, normloss.data[0]


def train_label(net, sample, regOptimizer, criterion):

    # labeled data
    inputImgs = sample['img']
    labels = sample['label']
    inputState = Variable(inputImgs,requires_grad=True)
    targetreg = Variable(labels,requires_grad=False)

    inputState = inputState.cuda()
    targetreg = targetreg.cuda()

    # forward + backward + optimize
    output = net(inputState)
    loss_label = criterion(output, targetreg)

    # zero the parameter gradients
    regOptimizer.zero_grad()
    loss_label.backward()
    regOptimizer.step()

    return loss_label.data[0]


def test_label(val_sample, net, criterion):
    # test on valset/trainset
    # return loss and accuracy
    ind = 0
    # mean_acc = 0.0
    # lossplot = []
    inputImgs = val_sample['img']
    labels = val_sample['label']
    inputState = Variable(inputImgs,requires_grad=False)
    targetreg = Variable(labels,requires_grad=False)
    inputState = inputState.cuda()
    targetreg = targetreg.cuda()

    output = net(inputState)
    loss = criterion(output, targetreg)
    val_loss = loss.data[0]

    return val_loss #, mean_acc/batchnum

labellossplot = []
vallossplot1 = []
vallossplot2 = []
unlabellossplot = []
# running_acc = 0.0
val_loss1 = 0.0
val_loss2 = 0.0
ind = 0

dataiter = iter(dataloader)
valiter1 = iter(valloader1)
valiter2 = iter(valloader2)
unlabeliter = iter(unlabelloader)
while True:

    ind += 1

    try:
        sample = dataiter.next()
    except:
        dataiter = iter(dataloader)
        sample = dataiter.next()

    try:
        unlabel_sample = unlabeliter.next()
    except:
        unlabeliter = iter(unlabelloder)
        unlabel_sample = unlabeliter.next()


    # label_loss = train_label(net, sample, regOptimizer, criterion)
    label_loss, unlabel_loss, add_loss = train_label_unlabel(net, sample, unlabel_sample, regOptimizer, criterion, lamb)
    labellossplot.append(label_loss)
    unlabellossplot.append(unlabel_loss)

    if ind % showiter == 0:    # print every 20 mini-batches
        try:
            val_sample1 = valiter1.next()
        except:
            valiter1 = iter(valloader1)
            val_sample1 = valiter1.next()
        val_loss1 = test_label(val_sample1, net, criterion)
        
        try:
            val_sample2 = valiter2.next()
        except:
            valiter2 = iter(valloader2)
            val_sample2 = valiter2.next()
        val_loss2 = test_label(val_sample2, net, criterion)

        vallossplot1.append(val_loss1)
        vallossplot2.append(val_loss2)

    print('[%s %d] loss: %.5f, val-loss: %.5f, val-loss: %.5f' %
        (exp_prefix[:-1], ind , label_loss ,val_loss1, val_loss2))

    if (ind)%snapshot==0:
        torch.save(net.state_dict(), predictModel+'_'+str(ind)+'.pkl')
        np.save(join(datadir,exp_prefix+'lossplot.npy'), labellossplot)
        np.save(join(datadir,exp_prefix+'vallossplot1.npy'), vallossplot1)
        np.save(join(datadir,exp_prefix+'vallossplot2.npy'), vallossplot2)
        np.save(join(datadir,exp_prefix+'unlabellossplot.npy'), unlabellossplot)
        
    if ind==trainstep:
        break
    


import matplotlib.pyplot as plt
ax1 = plt.subplot(121)
labellossplot = np.array(labellossplot)
labellossplot = labellossplot.reshape((-1,1))
labellossplot = labellossplot.mean(axis=1)
ax1.plot(labellossplot)

vallossplot1 = np.array(vallossplot1)
ax1.plot(range(showiter,trainstep+1,showiter),vallossplot1)
ax1.grid()

ax2 = plt.subplot(122)
labellossplot = np.array(labellossplot)
labellossplot = labellossplot.reshape((-1,1))
labellossplot = labellossplot.mean(axis=1)
ax2.plot(labellossplot)

vallossplot2 = np.array(vallossplot2)
ax2.plot(range(showiter,trainstep+1,showiter),vallossplot2)
ax2.grid()

plt.savefig(join(imgoutdir, predictModel.split('/')[-1]+'.png'))
# plt.ylim([0,1])
plt.show()
import ipdb; ipdb.set_trace()

