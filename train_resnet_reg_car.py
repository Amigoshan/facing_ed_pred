# This file is modified from train_semi_reg_jointloss.py
# Change the feature extractor to resnet18
# change the normalization
# 

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
from koperLabelDataset import KoperLabelDataset
from trackingLabelData import TrackingLabelDataset
from gascolaDataset import GascolaDataset
from ResnetRegNet import ResnetReg
from StateEncoderDecoder import EncoderReg


exp_prefix = '32_3_'
# preTrainModel = 'models_facing/8_12_2_ed_reg_3000.pkl'
# preTrainModel = 'models_facing/3_5_ed_cls_10000.pkl'
preTrainModel = 'models_facing/32_1_encoder_decoder_facing_leaky_50000.pkl'
modelname = 'models_facing/'+exp_prefix+'resnet_reg'
imgoutdir = 'resimg_facing'
datadir = 'data_facing'
datasetdir = '/datasets'
Lr_label = 0.0005 
batch = 32
trainstep = 20000
showiter = 50
snapshot = 2000
train_layer_num = 0
model = 'ed' # 'resnet'

if model == 'resnet':
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    encoderReg = ResnetReg()
    encoderReg.cuda()

elif model == 'ed':
    mean=[0, 0, 0]
    std=[1, 1, 1]
    hiddens = [3,16,32,32,64,64,128,256] 
    kernels = [4,4,4,4,4,4,3]
    paddings = [1,1,1,1,1,1,0]
    strides = [2,2,2,2,2,2,1]
    encoderReg = EncoderReg(hiddens, kernels, strides, paddings, actfunc='leaky')
    loadPretrain2(encoderReg, preTrainModel)
    encoderReg.cuda()
   
else:
    print 'unknown model:', model

paramlist = list(encoderReg.parameters())
criterion = nn.MSELoss()

# import ipdb;ipdb.set_trace()
# kl_criterion = nn.KLDivLoss()
# regOptimizer = optim.SGD(predictNet.parameters(), lr = Lr, momentum=0.9)
regOptimizer = optim.Adam(paramlist[-train_layer_num:], lr = Lr_label)

trackingLabelDataset = TrackingLabelDataset(csv_file='/datadrive/data/aayush/combined_data2/train/annotations/car_annotations.csv',
                                             data_aug=True,mean=mean,std=std)
titsLabelDataset = FacingDroneLabelDataset(imgdir='/datadrive/TITS2016WangOnRoadVehicle/labeled', 
                                            data_aug=True,mean=mean,std=std)
koperLabelDataset = KoperLabelDataset(data_aug = True,mean=mean,std=std)

valset = GascolaDataset(data_aug=False,mean=mean,std=std)

valnum = 100
dataloader = DataLoader(trackingLabelDataset, batch_size=batch, shuffle=True, num_workers=2)
dataloader2 = DataLoader(titsLabelDataset, batch_size=batch, shuffle=True, num_workers=2)
dataloader3 = DataLoader(koperLabelDataset, batch_size=batch, shuffle=True, num_workers=2)
valloader = DataLoader(valset, batch_size=batch, shuffle=False, num_workers=2)
# unlabelloder = DataLoader(unlabelset, batch_size=1, shuffle=True, num_workers=2)

def train_label(encoderReg, sample, regOptimizer, criterion):

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

    # zero the parameter gradients
    regOptimizer.zero_grad()
    # loss_label.backward()
    loss_label.backward()
    regOptimizer.step()

    return loss_label.data[0]


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
    # for ind1 in range(unlabel_batch-1):
    #     for ind2 in range(ind1+1, unlabel_batch):
    #         w = abs(ind1 - ind2)
    #         wei = exp(-alpha*w)

    #         diff = (output_unlabel[ind1]-output_unlabel[ind2])*(output_unlabel[ind1]-output_unlabel[ind2])
    #         diff = diff.sum()/2.0
    #         loss_unlabel = loss_unlabel + (diff-thresh).clamp(0) * wei
    #         if wei<1e-2: # skip far away pairs
    #             break

    # loss = encode.sum()
    loss = loss_label + loss_unlabel * lamb #+ normloss * lamb2

    # zero the parameter gradients
    regOptimizer.zero_grad()
    # loss_label.backward()
    loss.backward()
    regOptimizer.step()

    return loss_label.data[0], loss_unlabel.data[0], loss.data[0]#, normloss.data[0]


def test_label(dataloader, encoderReg, criterion, batchnum = 1):
    # test on valset/trainset
    # return loss and accuracy
    ind = 0
    mean_loss = 0.0
    # mean_acc = 0.0
    # lossplot = []
    for val_sample in dataloader:
        ind += 1
        inputImgs = val_sample['img']
        labels = val_sample['label']
        inputState = Variable(inputImgs,requires_grad=False)
        targetreg = Variable(labels,requires_grad=False)
        inputState = inputState.cuda()
        targetreg = targetreg.cuda()

        output, _ = encoderReg(inputState)
        loss = criterion(output, targetreg)
        val_loss = loss.data[0]
        mean_loss += val_loss

        if ind == batchnum:
            break

    return mean_loss/batchnum #, mean_acc/batchnum


lossplot = []
labellossplot = []
unlabellossplot = []
vallossplot = []
# running_acc = 0.0
val_loss = 0.0
# val_acc = 0.0
unlabel_loss = 0.0
norm_loss = 0.0
ind = 0

dataiter = iter(dataloader)
dataiter2 = iter(dataloader2)
dataiter3 = iter(dataloader3)
dataiter4 = iter(valloader)
# unlabeliter = iter(unlabelloder)
while True:

    ind += 1

    if ind%30==0:
        try:
            sample = dataiter4.next()
        except:
            dataiter4 = iter(valloader)
            sample = dataiter4.next()
    elif ind%3==0:
        try:
            sample = dataiter.next()
        except:
            dataiter = iter(dataloader)
            sample = dataiter.next()
    elif ind%3==1:
        try:
            sample = dataiter2.next()
        except:
            dataiter2 = iter(dataloader2)
            sample = dataiter2.next()
       
    else:
        try:
            sample = dataiter3.next()
        except:
            dataiter3 = iter(dataloader3)
            sample = dataiter3.next()
       

    # try:
    #     unlabel_sample = unlabeliter.next()
    # except:
    #     unlabeliter = iter(unlabelloder)
    #     unlabel_sample = unlabeliter.next()

    label_loss = train_label(encoderReg, sample, regOptimizer, criterion)
    labellossplot.append(label_loss)

    if ind % showiter == 0:    # print every 20 mini-batches
        val_loss = test_label(valloader, encoderReg, criterion, batchnum = 1)
    # train_loss, train_acc = test_label(dataloader, encoderReg, criterion, batchnum = 3)
    vallossplot.append(val_loss)

    print('[%s %d] loss: %.5f, val-loss: %.5f' %
        (exp_prefix[:-1], ind, label_loss ,val_loss))

    if (ind)%snapshot==0:
        torch.save(encoderReg.state_dict(), modelname+'_'+str(ind)+'.pkl')
        np.save(join(datadir,exp_prefix+'labellossplot.npy'), labellossplot)
        np.save(join(datadir,exp_prefix+'vallossplot.npy'), vallossplot)

    if ind==trainstep:
        break
    


import matplotlib.pyplot as plt
labellossplot = np.array(labellossplot)
labellossplot = labellossplot.reshape((-1,1))
labellossplot = labellossplot.mean(axis=1)
plt.plot(labellossplot)

vallossplot = np.array(vallossplot)
plt.plot(vallossplot)
plt.grid()

plt.savefig(join(imgoutdir, modelname.split('/')[-1]+'.png'))
# plt.ylim([0,1])
plt.show()
import ipdb; ipdb.set_trace()

