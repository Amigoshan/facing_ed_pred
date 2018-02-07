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

from utils import loadPretrain2, loadPretrain, groupPlot
from facingDroneLabelData import FacingDroneLabelDataset
from facingDroneUnlabelData import FacingDroneUnlabelDataset
from facingLabelData import FacingLabelDataset
from StateEncoderDecoder import EncoderReg


exp_prefix = '7_11_'
# preTrainModel = 'models_facing/5_2_ed_reg_2000.pkl'
# preTrainModel = 'models_facing/3_5_ed_cls_10000.pkl'
preTrainModel = 'models_facing/1_2_encoder_decoder_facing_leaky_50000.pkl'
predictModel = 'models_facing/'+exp_prefix+'ed_reg'
imgoutdir = 'resimg_facing'
Lr_label = 0.002 
Lr_unlabel = 0.001
batch = 32
trainstep = 5000
showiter = 10
snapshot = 1000
unlabel_batch = 32
lamb = 10 
alpha = 0.2
thresh = 0.01
train_layer_num = 8
label_train_num = 1
unlabel_train_num = 4

hiddens = [3,16,32,32,64,64,128,256] 
kernels = [4,4,4,4,4,4,3]
paddings = [1,1,1,1,1,1,0]
strides = [2,2,2,2,2,2,1]

encoderReg = EncoderReg(hiddens, kernels, strides, paddings, actfunc='leaky')
# encode the input using pretrained model
print 'load pretrained...'
encoderReg=loadPretrain(encoderReg,preTrainModel)
encoderReg.cuda()


paramlist = list(encoderReg.parameters())
criterion = nn.MSELoss()
# regOptimizer = optim.SGD(predictNet.parameters(), lr = Lr, momentum=0.9)
regOptimizer = optim.Adam(paramlist[-train_layer_num:], lr = Lr_label)
unlabelOptimizer = optim.Adam(paramlist[-train_layer_num:], lr = Lr_unlabel)

imgdataset = FacingDroneLabelDataset()
valset = FacingDroneLabelDataset(imgdir='/datasets/droneData/val')
# valset = FacingLabelDataset()
# imgdataset = FacingLabelDataset()
unlabelset = FacingDroneUnlabelDataset(batch = unlabel_batch, data_aug=True)

valnum = 100
dataloader = DataLoader(imgdataset, batch_size=batch, shuffle=True, num_workers=4)
valloader = DataLoader(valset, batch_size=valnum, shuffle=False, num_workers=8)
unlabelloder = DataLoader(unlabelset, batch_size=1, shuffle=True, num_workers=2)

def train_unlabel(encoderReg, unlabelloder, unlabelOptimizer, step):
    k = 0
    unlabel_loss = 0
    # norm_loss = 0
    for sample in unlabelloder:
        k = k + 1

        imgseq = sample.squeeze()
        inputState = Variable(imgseq,requires_grad=True).cuda()

        inputState = inputState.cuda()

        # zero the parameter gradients
        unlabelOptimizer.zero_grad()

        # import ipdb; ipdb.set_trace()

        # forward + backward + optimize
        output, _ = encoderReg(inputState)
        # outtensor = output.data
        # clear the loss of those unlabeled samples
        # this needs batch>1

        # norm = output * output
        # norm = norm.sum(dim=1)
        # one_var = Variable(torch.ones(output.size()[0])).cuda()
        # normloss = (norm - one_var).abs().sum()

        # for ind in range(unlabel_batch):
        #     norm1 = (output[ind]*output[ind]).sum()
        #     norm += (1-norm1).abs()

        loss_unlabel = Variable(torch.Tensor([0])).cuda()
        for ind1 in range(unlabel_batch-1):
            for ind2 in range(ind1+1, unlabel_batch):
                w = abs(ind1 - ind2)
                wei = exp(-alpha*w)

                diff = (output[ind1]-output[ind2])*(output[ind1]-output[ind2])
                diff = diff.sum()/2.0
                loss_unlabel = loss_unlabel + (diff-thresh).clamp(0) * wei
                if wei<1e-2: # skip far away pairs
                    break
        # loss_unlabel += norm
        # normloss.backward(retain_graph=True)
        loss_unlabel.backward()
        unlabelOptimizer.step()

        # losslist.append(loss_unlabel.data[0])
        # norm_loss += normloss.data[0]
        unlabel_loss += loss_unlabel.data[0]
        if k==step:
            break
    return unlabel_loss/step #, norm_loss/step


def train_label(encoderReg, dataloader, regOptimizer, criterion, step):
    k = 0
    running_loss = 0
    mean_acc = 0
    for sample in dataloader:  # loop over the dataset multiple times
        k = k + 1
        inputImgs = sample['img']
        labels = sample['label']
        # print inputImgs.size(), outputImgs.size(),actions.size()

        # print 'imput img shape:', inputImg.shape
        inputState = Variable(inputImgs,requires_grad=True)
        targetreg = Variable(labels,requires_grad=False)

        inputState = inputState.cuda()
        targetreg = targetreg.cuda()

        # zero the parameter gradients
        regOptimizer.zero_grad()

        # import ipdb; ipdb.set_trace()

        # forward + backward + optimize
        output, _ = encoderReg(inputState)
        loss = criterion(output, targetreg)

        loss.backward()
        regOptimizer.step()


        # losslist.append(loss.data[0])
        running_loss += loss.data[0]

        if k == step:
            break

    return running_loss/step #, mean_acc/step


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
        # lossplot.append(val_loss)

        # _, pred = output.topk(1, 1, True, True)

        # correct = pred.squeeze().eq(targetreg)
        # val_acc = correct.view(-1).float().sum(0)
        # val_acc = val_acc/labels.size()[0]
        # mean_acc += val_acc

        if ind == batchnum:
            break

    return mean_loss/batchnum #, mean_acc/batchnum


lossplot = []
unlabellossplot = []
vallossplot = []
running_loss = 0.0
# running_acc = 0.0
val_loss = 0.0
# val_acc = 0.0
unlabel_loss = 0.0
# norm_loss = 0.0
ind = 0
while True:

    ind += label_train_num + unlabel_train_num

    running_loss = train_label(encoderReg, dataloader, regOptimizer, criterion, label_train_num)
    lossplot.append(running_loss)

    unlabel_loss = train_unlabel(encoderReg, unlabelloder, unlabelOptimizer, unlabel_train_num)
    unlabellossplot.append(unlabel_loss)

    # if ind % showiter == 0:    # print every 20 mini-batches
    val_loss = test_label(valloader, encoderReg, criterion, batchnum = 1)
    # train_loss, train_acc = test_label(dataloader, encoderReg, criterion, batchnum = 3)
    vallossplot.append(val_loss)

    print('[%s %d] loss: %.5f, val-loss: %.5f, unlabel-loss: %.5f' %
    (exp_prefix[:-1], ind , running_loss ,val_loss, unlabel_loss))

    if (ind)%snapshot==0:
        torch.save(encoderReg.state_dict(), predictModel+'_'+str(ind)+'.pkl')

    if ind==trainstep:
        break
    


import matplotlib.pyplot as plt
ax1 = plt.subplot(121)
lossplot = np.array(lossplot)
lossplot = lossplot.reshape((-1,1))
lossplot = lossplot.mean(axis=1)
ax1.plot(lossplot)

vallossplot = np.array(vallossplot)
ax1.plot(vallossplot)
ax1.grid()

ax2 = plt.subplot(122)
unlabellossplot = np.array(unlabellossplot)
gpunlabelx, gpunlabely = groupPlot(range(len(unlabellossplot)),unlabellossplot)
ax2.plot(unlabellossplot)
ax2.plot(gpunlabelx, gpunlabely, color='y')
ax2.grid()

plt.savefig(join(imgoutdir, predictModel.split('/')[-1]+'.png'))
# plt.ylim([0,1])
plt.show()
import ipdb; ipdb.set_trace()

