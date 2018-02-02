import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from os.path import join
from math import exp

from utils import loadPretrain2, loadPretrain
from facingDroneLabelData import FacingDroneLabelDataset
from facingDroneUnlabelData import FacingDroneUnlabelDataset
from facingLabelData import FacingLabelDataset
from StateEncoderDecoder import EncoderCls
from Predictor import PredictNet

np.set_printoptions(threshold=np.nan, precision=2, suppress=True)

preTrainModel = 'models_facing/2_6_ed_cls_1000.pkl'
batch = 32
unlabel_batch = 32

hiddens = [3,16,32,32,64,64,128,256] 
kernels = [4,4,4,4,4,4,3]
paddings = [1,1,1,1,1,1,0]
strides = [2,2,2,2,2,2,1]

encoderCls = EncoderCls(hiddens, kernels, strides, paddings, actfunc='leaky')
# encode the input using pretrained model
print 'load pretrained...'
encoderCls=loadPretrain(encoderCls,preTrainModel)
encoderCls.cuda()

criterion = nn.CrossEntropyLoss()

imgdataset = FacingDroneLabelDataset()
valset = FacingDroneLabelDataset(imgdir='/home/wenshan/datasets/droneData/val')
# valset = FacingLabelDataset()
# imgdataset = FacingLabelDataset()
unlabelset = FacingDroneUnlabelDataset(batch = unlabel_batch)

valnum = 100
dataloader = DataLoader(imgdataset, batch_size=batch, shuffle=True, num_workers=4)
valloader = DataLoader(valset, batch_size=valnum, shuffle=False, num_workers=8)
unlabelloder = DataLoader(unlabelset, batch_size=1, shuffle=True, num_workers=2)


def test_label(dataloader, encoderCls, criterion, display=True):
    # test on valset/trainset
    # return loss and accuracy
    ind = 0
    mean_loss = 0.0
    mean_acc = 0.0
    # lossplot = []
    for val_sample in dataloader:
        ind += 1
        inputImgs = val_sample['img']
        labels = val_sample['label']
        targetCls = Variable(labels,requires_grad=False).cuda()

        inputState = Variable(inputImgs,requires_grad=False).cuda()

        output, _ = encoderCls(inputState)
        loss = criterion(output, targetCls)
        val_loss = loss.data[0]
        mean_loss += val_loss

        _, pred = output.topk(1, 1, True, True)

        correct = pred.squeeze().eq(targetCls)
        val_acc = correct.view(-1).float().sum(0)
        val_acc = val_acc/valnum
        mean_acc += val_acc

        print pred.data.cpu().squeeze().numpy()
        print val_sample['label'].numpy()
        if display:
            unlabelset.seq_show(val_sample['img'].numpy())

        break

    return mean_loss, mean_acc


def test_unlabel(dataloader, encoderCls, display=True):
    # test on valset/trainset
    # return the output on one batch
    ind = 0
    mean_loss = 0.0
    mean_acc = 0.0
    # lossplot = []
    for val_sample in dataloader:
        ind += 1
        inputImgs = val_sample.squeeze()

        inputState = Variable(inputImgs,requires_grad=False).cuda()

        output, _ = encoderCls(inputState)

        break

    _, pred = output.topk(1, 1, True, True)
    print pred.data.cpu().squeeze().numpy()
    if display:
        unlabelset.seq_show(val_sample.squeeze().numpy())
    return pred.squeeze()



for k in range(100):  # loop over the dataset multiple times

    test_label(dataloader, encoderCls, criterion, display = True)
    pred = test_unlabel(unlabelloder, encoderCls, display=True)
