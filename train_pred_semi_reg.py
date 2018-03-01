# This file is modified from train_ed_semi_cls.py
# Change the classification model to regression model

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
from facingLabelData import FacingLabelDataset
from StateEncoderDecoder import EncoderReg_Pred


exp_prefix = '20_2_'
# preTrainModel = 'models_facing/8_13_ed_reg_46000.pkl'
# preTrainModel = 'models_facing/3_5_ed_cls_10000.pkl'
# preTrainModel = 'models_facing/1_2_encoder_decoder_facing_leaky_50000.pkl'
# preTrainModel = 'models_facing/13_1_ed_reg_100000.pkl'
predictModel = 'models_facing/'+exp_prefix+'pred_reg'
imgoutdir = 'resimg_facing'
datadir = 'data_facing'
datasetdir = '/home/wenshan/datasets'
lr = 0.01
lamb = 5.0
unlabel_batch = 32
batch = 32
trainstep = 10000
showiter = 50
snapshot = 2000
train_layer_num = 0

hiddens = [3,16,32,32,64,64,128,256] 
kernels = [4,4,4,4,4,4,3]
paddings = [1,1,1,1,1,1,0]
strides = [2,2,2,2,2,2,1]

encoderReg = EncoderReg_Pred(hiddens, kernels, strides, paddings, actfunc='leaky', rnnHidNum=128)
# encode the input using pretrained model
# print 'load pretrained...'
# encoderReg=loadPretrain(encoderReg,preTrainModel)
encoderReg.cuda()


paramlist = list(encoderReg.parameters())
criterion = nn.MSELoss()
regOptimizer = optim.SGD(paramlist[-train_layer_num:], lr = lr, momentum=0.9)
# regOptimizer = optim.Adam(paramlist[-train_layer_num:], lr = lr)

imgdataset = FacingDroneLabelDataset(imgdir=join(datasetdir,'droneData/label'), data_aug=True)
valset = FacingDroneLabelDataset(imgdir=join(datasetdir,'droneData/val'))
imgdataset2 = FacingLabelDataset(annodir = join(datasetdir,'facing/facing_anno'), 
                                 imgdir=join(datasetdir,'facing/facing_img_coco'), 
                                 data_aug=True)
# imgdataset3 = TrackingLabelDataset(data_aug=True)
# imgdataset = FacingLabelDataset()
unlabelset = FacingDroneUnlabelDataset(imgdir=join(datasetdir,'dirimg'), 
                                       batch = unlabel_batch, data_aug=True, extend=True)
# unlabelset2 = TrackingUnlabelDataset(batch = unlabel_batch, data_aug=True)


valnum = 100
dataloader = DataLoader(imgdataset, batch_size=batch, shuffle=True, num_workers=2)
dataloader2 = DataLoader(imgdataset2, batch_size=batch, shuffle=True, num_workers=2)
valloader = DataLoader(valset, batch_size=valnum, shuffle=False, num_workers=2)
unlabelloder = DataLoader(unlabelset, batch_size=1, shuffle=True, num_workers=2)

def train_label_unlabel(encoderReg, sample, unlabel_sample, regOptimizer, criterion, lamb):

    # loss_label = Variable(torch.Tensor([0])).cuda()
    # loss = Variable(torch.Tensor([0])).cuda()
    # loss_pred = Variable(torch.Tensor([0])).cuda()
    # labeled data
    inputImgs = sample['img']
    labels = sample['label']
    inputState = Variable(inputImgs,requires_grad=True)
    targetreg = Variable(labels,requires_grad=False)

    inputState = inputState.cuda()
    targetreg = targetreg.cuda()

    # forward + backward + optimize
    output, _, _ = encoderReg(inputState)
    loss_label = criterion(output, targetreg)

    # unlabeled data
    imgseq = unlabel_sample.squeeze()
    inputState_unlabel = Variable(imgseq,requires_grad=True).cuda()
    inputState_unlabel = inputState_unlabel.cuda()

    # import ipdb; ipdb.set_trace()

    output_unlabel, encode, pred = encoderReg(inputState_unlabel)
    pred_target = encode[unlabel_batch/2:,:].detach()
    # print np.mean(encode.data.cpu().numpy()),np.std(encode.data.cpu().numpy())

    loss_pred = criterion(pred, pred_target)

    loss = loss_label + loss_pred * lamb #+ normloss * lamb2

    # zero the parameter gradients
    regOptimizer.zero_grad()
    # loss_label.backward()
    # loss_pred.backward()
    loss.backward()
    regOptimizer.step()

    return loss_label.data[0], loss_pred.data[0], loss.data[0]#, normloss.data[0]

def test_label(val_sample, encoderReg, criterion, batchnum = 1):

    inputImgs = val_sample['img']
    labels = val_sample['label']
    inputState = Variable(inputImgs,requires_grad=False)
    targetreg = Variable(labels,requires_grad=False)
    inputState = inputState.cuda()
    targetreg = targetreg.cuda()

    output, _, _ = encoderReg(inputState)
    loss = criterion(output, targetreg)
    val_loss = loss.data[0]

    return val_loss #, mean_acc/batchnum


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
add_loss = 0.0
label_loss = 0.0

dataiter = iter(dataloader)
dataiter2 = iter(dataloader2)
unlabeliter = iter(unlabelloder)
while True:

    ind += 1

    if ind%2==0:
        try:
            sample = dataiter.next()
        except:
            dataiter = iter(dataloader)
            sample = dataiter.next()
    else:
        try:
            sample = dataiter2.next()
        except:
            dataiter2 = iter(dataloader2)
            sample = dataiter2.next()
       

    try:
        unlabel_sample = unlabeliter.next()
    except:
        unlabeliter = iter(unlabelloder)
        unlabel_sample = unlabeliter.next()

    label_loss, unlabel_loss, add_loss = train_label_unlabel(encoderReg, sample, unlabel_sample, regOptimizer, criterion, lamb)
    lossplot.append(add_loss)
    labellossplot.append(label_loss)
    unlabellossplot.append(unlabel_loss)

    if (ind-1) % showiter == 0:    # print every 20 mini-batches
        # val_loss = test_label(valloader, encoderReg, criterion, batchnum = 1)
        for val_sample in valloader:
            # print val_sample['img'].size()
            val_loss = test_label(val_sample, encoderReg, criterion)
    # train_loss, train_acc = test_label(dataloader, encoderReg, criterion, batchnum = 3)
    vallossplot.append(val_loss)

    print('[%s %d] loss: %.5f, label-loss: %.5f, val-loss: %.5f, unlabel-loss: %.5f' %
        (exp_prefix[:-1], ind , add_loss, label_loss ,val_loss, unlabel_loss))

    if (ind)%snapshot==0:
        torch.save(encoderReg.state_dict(), predictModel+'_'+str(ind)+'.pkl')
        np.save(join(datadir,exp_prefix+'lossplot.npy'), lossplot)
        np.save(join(datadir,exp_prefix+'vallossplot.npy'), vallossplot)
        np.save(join(datadir,exp_prefix+'unlabellossplot.npy'), unlabellossplot)
        np.save(join(datadir,exp_prefix+'labellossplot.npy'), labellossplot)
    if ind==trainstep:
        break
    


import matplotlib.pyplot as plt
ax1 = plt.subplot(131)
labellossplot = np.array(labellossplot)
labellossplot = labellossplot.reshape((-1,1))
labellossplot = labellossplot.mean(axis=1)
ax1.plot(labellossplot)

vallossplot = np.array(vallossplot)
ax1.plot(vallossplot)
ax1.grid()

ax2 = plt.subplot(132)
lossplot = np.array(lossplot)
lossplot = lossplot.reshape((-1,1))
lossplot = lossplot.mean(axis=1)
ax2.plot(lossplot)
ax2.grid()

ax3 = plt.subplot(133)
unlabellossplot = np.array(unlabellossplot)
gpunlabelx, gpunlabely = groupPlot(range(len(unlabellossplot)),unlabellossplot)
ax3.plot(unlabellossplot)
ax3.plot(gpunlabelx, gpunlabely, color='y')
ax3.grid()

plt.savefig(join(imgoutdir, predictModel.split('/')[-1]+'.png'))
# plt.ylim([0,1])
plt.show()
import ipdb; ipdb.set_trace()

