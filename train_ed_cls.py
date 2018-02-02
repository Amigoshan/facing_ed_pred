import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from os.path import join

from utils import loadPretrain2
from facingDroneLabelData import FacingDroneLabelDataset
from facingLabelData import FacingLabelDataset
from StateEncoderDecoder import EncoderCls
from Predictor import PredictNet


exp_prefix = '2_6_'
preTrainModel = 'models_facing/1_2_encoder_decoder_facing_leaky_50000.pkl'
predictModel = 'models_facing/'+exp_prefix+'ed_cls'
imgoutdir = 'resimg_facing'
Lr = 0.001 
batch = 32
trainstep = 1000
showiter = 10
snapshot = 200

hiddens = [3,16,32,32,64,64,128,256] 
kernels = [4,4,4,4,4,4,3]
paddings = [1,1,1,1,1,1,0]
strides = [2,2,2,2,2,2,1]

encoderCls = EncoderCls(hiddens, kernels, strides, paddings, actfunc='leaky')
# encode the input using pretrained model
print 'load pretrained...'
encoderCls=loadPretrain2(encoderCls,preTrainModel)
encoderCls.cuda()


paramlist = list(encoderCls.parameters())
criterion = nn.CrossEntropyLoss()
# clsOptimizer = optim.SGD(predictNet.parameters(), lr = Lr, momentum=0.9)
clsOptimizer = optim.Adam(paramlist[-2:], lr = Lr)

imgdataset = FacingDroneLabelDataset()
valset = FacingDroneLabelDataset(imgdir='/home/wenshan/datasets/droneData/val')
# valset = FacingLabelDataset()
# imgdataset = FacingLabelDataset()

valnum = 100
dataloader = DataLoader(imgdataset, batch_size=batch, shuffle=True, num_workers=4)
valloader = DataLoader(valset, batch_size=valnum, shuffle=False, num_workers=8)

lossplot = []
vallossplot = []
running_loss = 0.0
running_acc = 0.0
val_loss = 0.0
val_acc = 0.0
ind = 0
while True:
    for sample in dataloader:  # loop over the dataset multiple times
        ind = ind + 1
        inputImgs = sample['img']
        labels = sample['label']
        # print inputImgs.size(), outputImgs.size(),actions.size()

        # print 'imput img shape:', inputImg.shape
        inputState = Variable(inputImgs,requires_grad=True)
        targetCls = Variable(labels,requires_grad=False)

        inputState = inputState.cuda()
        targetCls = targetCls.cuda()

        # zero the parameter gradients
        # stateOptimizer.zero_grad()
        clsOptimizer.zero_grad()

        # import ipdb; ipdb.set_trace()

        # forward + backward + optimize
        output, _ = encoderCls(inputState)
        loss = criterion(output, targetCls)

        lossplot.append(loss.data[0])

        loss.backward()
        clsOptimizer.step()


        running_loss += loss.data[0]
        if ind % showiter == 0:    # print every 20 mini-batches

            # test on valset
            for val_sample in valloader:
                inputImgs = val_sample['img']
                labels = val_sample['label']
                inputState = Variable(inputImgs,requires_grad=False)
                targetCls = Variable(labels,requires_grad=False)
                inputState = inputState.cuda()
                targetCls = targetCls.cuda()

                output, _ = encoderCls(inputState)
                loss = criterion(output, targetCls)
                val_loss = loss.data[0]

                _, pred = output.topk(1, 1, True, True)

                correct = pred.squeeze().eq(targetCls)
                val_acc = correct.view(-1).float().sum(0)
                val_acc = val_acc/valnum
                vallossplot.append([ind,val_loss])

                break

            print('[%d] loss: %.5f lr: %f, val-loss: %.5f, val-acc: %.5f' %
            (ind , running_loss / showiter, clsOptimizer.param_groups[0]['lr'], val_loss, val_acc))
            running_loss = 0.0

        if (ind)%snapshot==0:
            torch.save(encoderCls.state_dict(), predictModel+'_'+str(ind)+'.pkl')

        # if (ind)%30000==0 or (ind)%40000==0:
        #   Lr = Lr * 0.2
        #   for param_group in clsOptimizer.param_groups:
        #       param_group['lr'] = Lr
        if ind==trainstep:
            break

    if ind==trainstep:
        break
    


import matplotlib.pyplot as plt
lossplot = np.array(lossplot)
lossplot = lossplot.reshape((-1,1))
lossplot = lossplot.mean(axis=1)
plt.plot(lossplot)
vallossplot = np.array(vallossplot)
plt.plot(vallossplot[:,0],vallossplot[:,1])
plt.grid()
plt.savefig(join(imgoutdir, predictModel.split('/')[-1]+'.png'))
# plt.ylim([0,1])
plt.show()
import ipdb; ipdb.set_trace()

