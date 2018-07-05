import cv2
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from os.path import join

from utils import loadPretrain2, loadPretrain, seq_show_with_arrow
# from facingDroneLabelData import FacingDroneLabelDataset
# from facingDroneUnlabelData import FacingDroneUnlabelDataset
# from facingLabelData import FacingLabelDataset
from gascolaDataset import GascolaDataset
from ResnetRegNet import ResnetReg
from StateEncoderDecoder import EncoderReg
from cocoData import CocoDataset

np.set_printoptions(threshold=np.nan, precision=2, suppress=True)

batch = 8
unlabel_batch = 32
# datasetdir = '/datasets'
model = 'resnet' # 'resnet'

criterion = nn.MSELoss()

def test_label(sample, encoderReg, criterion, display=True, compute_loss=True):
    # test on valset/trainset
    # return loss and accuracy
    inputImgs = sample['img']
    inputState = Variable(inputImgs,requires_grad=False).cuda()
    output, _ = encoderReg(inputState)

    if display:
        print 'network output:', output.data
        seq_show_with_arrow(inputImgs.numpy(), output.data.cpu().numpy(), mean=mean,std=std)

    if compute_loss:
        labels = sample['label']
        targetCls = Variable(labels,requires_grad=False).cuda()
        loss = criterion(output, targetCls)
        val_loss = loss.data[0]
        mean_loss += val_loss
        print labels.numpy()
        print 'loss ',loss.data[0]

        return output, loss.data[0]

    return output


def test_unlabel(sample, encoderReg, display=True):
    # test on valset/trainset
    # return the output on one batch
    inputImgs = sample.squeeze()
    inputState = Variable(inputImgs,requires_grad=False).cuda()
    output, _ = encoderReg(inputState)

    if display:
        seq_show_with_arrow(inputImgs.numpy(), output.data.cpu().numpy(), scale = 0.5, mean=mean,std=std)
    return output


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
    encoderReg.cuda()
   
else:
    print 'unknown model:', model

# imgdataset = GascolaDataset(data_aug=False,mean=mean,std=std)
# dataloader = DataLoader(imgdataset, batch_size=batch, shuffle=True, num_workers=1)

cocodataset = CocoDataset(data_aug=False,mean=mean,std=std)
dataloader = DataLoader(cocodataset, batch_size=batch, shuffle=True, num_workers=1)

for k in range(10000,10001,2000):
    preTrainModel = 'models_facing/30_2_resnet_reg_car_'+str(k)+'.pkl'
    loadPretrain(encoderReg, preTrainModel)

    total_loss = 0
    for sample in dataloader:
        _ = test_label(sample, encoderReg, criterion, display = True, compute_loss=False)
    print preTrainModel.split('/')[-1]
    print '  total loss:', mean_loss

    # test_label(valloader, encoderReg, criterion, display = True)
    # test_label(cocoloader, encoderReg, criterion, display = True, compute_loss=False)
    # pred = test_unlabel(unlabelloder, encoderReg, display=True)
