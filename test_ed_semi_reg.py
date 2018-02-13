import cv2
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from os.path import join

from utils import loadPretrain2, loadPretrain, seq_show_with_arrow
from facingDroneLabelData import FacingDroneLabelDataset
from facingDroneUnlabelData import FacingDroneUnlabelDataset
from facingLabelData import FacingLabelDataset
from StateEncoderDecoder import EncoderReg

np.set_printoptions(threshold=np.nan, precision=2, suppress=True)

preTrainModel = 'models_facing/8_13_ed_reg_46000.pkl'
batch = 8
unlabel_batch = 32

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
unlabelset = FacingDroneUnlabelDataset(batch = unlabel_batch, data_aug=True)
cocodata = FacingLabelDataset()


valnum = 32
dataloader = DataLoader(imgdataset, batch_size=batch, shuffle=True, num_workers=1)
valloader = DataLoader(valset, batch_size=valnum, shuffle=True, num_workers=1)
unlabelloder = DataLoader(unlabelset, batch_size=1, shuffle=True, num_workers=1)
cocoloader = DataLoader(cocodata, batch_size=batch, shuffle=True, num_workers=1)

def test_label(dataloader, encoderReg, criterion, display=True, compute_loss=True):
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

        if compute_loss:
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
        seq_show_with_arrow(inputImgs.numpy(), output.data.cpu().numpy(), scale = 0.5)
    return output



for k in range(100):  # loop over the dataset multiple times

    test_label(dataloader, encoderReg, criterion, display = True)
    test_label(valloader, encoderReg, criterion, display = True)
    test_label(cocoloader, encoderReg, criterion, display = True, compute_loss=False)
    pred = test_unlabel(unlabelloder, encoderReg, display=True)
