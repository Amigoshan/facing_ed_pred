import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from os.path import join
from math import exp


from utils import loadPretrain2, loadPretrain, seq_show_with_arrow
from facingDroneLabelData import FacingDroneLabelDataset
from facingDroneUnlabelData import FacingDroneUnlabelDataset
from facingLabelData import FacingLabelDataset
from StateEncoderDecoder import EncoderReg

np.set_printoptions(threshold=np.nan, precision=2, suppress=True)

preTrainModel = 'models_facing/8_1_ed_reg_3000.pkl'
batch = 8
unlabel_batch = 8

hiddens = [3,16,32,32,64,64,128,256] 
kernels = [4,4,4,4,4,4,3]
paddings = [1,1,1,1,1,1,0]
strides = [2,2,2,2,2,2,1]

encoderReg = EncoderReg(hiddens, kernels, strides, paddings, actfunc='leaky')
# encode the input using pretrained model
print 'load pretrained...'
encoderReg=loadPretrain(encoderReg,preTrainModel)
encoderReg.cuda()

criterion = nn.MSELoss()

imgdataset = FacingDroneLabelDataset()
valset = FacingDroneLabelDataset(imgdir='/datasets/droneData/val')
# valset = FacingLabelDataset()
# imgdataset = FacingLabelDataset()
unlabelset = FacingDroneUnlabelDataset(batch = unlabel_batch, data_aug=True)

valnum = 32
dataloader = DataLoader(imgdataset, batch_size=batch, shuffle=True, num_workers=4)
valloader = DataLoader(valset, batch_size=valnum, shuffle=True, num_workers=8)
unlabelloder = DataLoader(unlabelset, batch_size=1, shuffle=True, num_workers=2)


def test_label(dataloader, encoderReg, criterion, display=True):
    # test on valset/trainset
    # return loss and accuracy
    ind = 0
    mean_loss = 0.0
    for val_sample in dataloader:
        ind += 1
        inputImgs = val_sample['img']
        labels = val_sample['label']
        targetCls = Variable(labels,requires_grad=False).cuda()

        inputState = Variable(inputImgs,requires_grad=False).cuda()

        output, _ = encoderReg(inputState)
        loss = criterion(output, targetCls)
        val_loss = loss.data[0]
        mean_loss += val_loss

        print labels.numpy()
        if display:
            seq_show_with_arrow(inputImgs.numpy(), output.data.cpu().numpy())

        break

    return output, mean_loss


def test_unlabel(dataloader, encoderReg, display=True):
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

        output, _ = encoderReg(inputState)

        break

    if display:
        seq_show_with_arrow(inputImgs.numpy(), output.data.cpu().numpy())
    return output



for k in range(100):  # loop over the dataset multiple times

    test_label(dataloader, encoderReg, criterion, display = True)
    test_label(valloader, encoderReg, criterion, display = True)
    pred = test_unlabel(unlabelloder, encoderReg, display=True)
