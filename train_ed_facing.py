# train an encoder-decoder using conv-deconv network
# train encoder-decoder on warped data

import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import math
import time
from os.path import join

# from utils import loadPretrain
from facingData import FacingDataset
from StateEncoderDecoder import StateEncoderDecoder

exp_prefix = '1_2_'
# LoadPretrain = False

exp_name = exp_prefix+'encoder_decoder_facing_leaky'
paramName = 'models_facing/'+ exp_name
imgoutdir = 'resimg_facing'
# preTrainModel = 'models/2_4_encoder_decoder_20000.pkl'
Lr = 0.001
batch = 32
trainstep = 50000
showiter = 100
snapshot = 10000

hiddens = [3,16,32,32,64,64,128,256] 
kernels = [4,4,4,4,4,4,3]
paddings = [1,1,1,1,1,1,0]
strides = [2,2,2,2,2,2,1]


stateEncoderDecoder = StateEncoderDecoder(hiddens, kernels, strides, paddings, actfunc='leaky')

# if LoadPretrain:
# 	stateEncoderDecoder=loadPretrain(stateEncoderDecoder,preTrainModel)

stateEncoderDecoder.cuda()

criterion = nn.SmoothL1Loss()
# stateOptimizer = optim.SGD(stateEncoderDecoder.parameters(), lr = Lr, momentum=0.9)
stateOptimizer = optim.Adam(stateEncoderDecoder.parameters(), lr = Lr) #,weight_decay=1e-5)

imgdataset = FacingDataset()
dataloader = DataLoader(imgdataset, batch_size=batch, shuffle=True, num_workers=4)

lossplot = []
running_loss = 0.0

ind = 0
while True:

	for sample in dataloader:  # loop over the dataset multiple times
		ind = ind + 1

		# inputImg = randomSample(map,inputSize)

		# print 'imput img shape:', inputImg.shape
		# inputTensor = torch.from_numpy(inputImg)
		inputState = Variable(sample.cuda(),requires_grad=True)
		targetState = Variable(sample.clone().cuda())
		# print inputState.size()

		# zero the parameter gradients
		stateOptimizer.zero_grad()

		# forward + backward + optimize
		output, _ = stateEncoderDecoder(inputState)
		loss = criterion(output, targetState)

		running_loss += loss.data[0]
		if ind % showiter == 0:    # print every 20 mini-batches
			timestr = time.strftime('%m/%d %H:%M:%S',time.localtime())
			print(exp_prefix[0:-1] + ' [%d %s] loss: %.5f lr: %f' %
			(ind , timestr, running_loss / showiter, Lr))
			running_loss = 0.0
		lossplot.append(loss.data[0])
		# print loss.data[0]

		loss.backward()
		stateOptimizer.step()

		if (ind)%snapshot==0:
			torch.save(stateEncoderDecoder.state_dict(), paramName+'_'+str(ind)+'.pkl')

		if ind==trainstep:
			break
	if ind==trainstep:
		break

# torch.save(stateEncoderDecoder.state_dict(), paramName)
	
import matplotlib.pyplot as plt
group = 10
lossplot = np.array(lossplot)
if len(lossplot)%group>0:
	lossplot = lossplot[0:len(lossplot)/group*group]
lossplot = lossplot.reshape((-1,group))
lossplot = lossplot.mean(axis=1)
plt.plot(lossplot)
plt.grid()
plt.savefig(join(imgoutdir, exp_name+'.png'))
plt.ylim([0,0.01])
plt.show()
import ipdb; ipdb.set_trace()

