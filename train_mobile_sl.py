# This file is modified from train_semi_reg_joinloss.py
# Change the mobilenet
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

# from utils import loadPretrain2, loadPretrain, groupPlot
from facingDroneLabelData import FacingDroneLabelDataset
# from facingDroneUnlabelData import FacingDroneUnlabelDataset
# from facingLabelData import FacingLabelDataset
from trackingLabelData import TrackingLabelDataset
# from trackingUnlabelData import TrackingUnlabelDataset
# from StateEncoderDecoder import EncoderReg_norm as EncoderReg
from MobileReg import MobileReg

exp_prefix = '51_2_'
preTrainModel = 'pretrained_models/mobilenet_v1_0.50_224.pth'
predictModel = 'models_facing/'+exp_prefix+'mobile_reg'
imgoutdir = 'resimg_facing'
datadir = 'data_facing'
datasetdir = '/datadrive/datasets'
Lr_label = 0.0005 
batch = 64
trainstep = 40000
showiter = 50
snapshot = 5000
train_layer_num = 12

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

net = MobileReg()
# encode the input using pretrained model
print 'load pretrained...'
net.load_pretrained_pth(preTrainModel)
net.cuda()


paramlist = list(net.parameters())
criterion = nn.MSELoss()
# kl_criterion = nn.KLDivLoss()
# regOptimizer = optim.SGD(predictNet.parameters(), lr = Lr, momentum=0.9)
regOptimizer = optim.Adam(paramlist[-train_layer_num:], lr = Lr_label)

valset1 = FacingDroneLabelDataset(imgdir=join(datasetdir,'droneData/val'),imgsize = 112,mean=mean,std=std)
valset2 = TrackingLabelDataset(filename='/datadrive/person/DukeMTMC/test_heading_gt.txt',imgsize = 112,mean=mean,std=std)
imgdataset = TrackingLabelDataset(filename='/datadrive/person/DukeMTMC/trainval_heading_gt.txt',imgsize = 112, data_aug=True,mean=mean,std=std)
valnum = 500
dataloader = DataLoader(imgdataset, batch_size=batch, shuffle=True, num_workers=2)
valloader1 = DataLoader(valset1, batch_size=valnum, shuffle=True, num_workers=2)
valloader2 = DataLoader(valset2, batch_size=valnum, shuffle=True, num_workers=2)

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
# running_acc = 0.0
val_loss1 = 0.0
val_loss2 = 0.0
ind = 0

dataiter = iter(dataloader)
valiter1 = iter(valloader1)
valiter2 = iter(valloader2)
while True:

    ind += 1

    try:
        sample = dataiter.next()
    except:
        dataiter = iter(dataloader)
        sample = dataiter.next()


    label_loss = train_label(net, sample, regOptimizer, criterion)
    labellossplot.append(label_loss)

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

    if ind==trainstep:
        break
    


import matplotlib.pyplot as plt
ax1 = plt.subplot(121)
labellossplot = np.array(labellossplot)
labellossplot = labellossplot.reshape((-1,1))
labellossplot = labellossplot.mean(axis=1)
ax1.plot(labellossplot)

vallossplot1 = np.array(vallossplot1)
ax1.plot(vallossplot1)
ax1.grid()

ax2 = plt.subplot(122)
labellossplot = np.array(labellossplot)
labellossplot = labellossplot.reshape((-1,1))
labellossplot = labellossplot.mean(axis=1)
ax2.plot(labellossplot)

vallossplot2 = np.array(vallossplot2)
ax2.plot(vallossplot2)
ax2.grid()

plt.savefig(join(imgoutdir, predictModel.split('/')[-1]+'.png'))
# plt.ylim([0,1])
plt.show()
import ipdb; ipdb.set_trace()

