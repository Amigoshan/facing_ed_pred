import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from os.path import join

from utils import loadPretrain
from mnistManipulationData import MnistManiDataset
from StateEncoderDecoder import StateEncoderDecoder
from Predictor import PredictNet


exp_prefix = '6_1_'
preTrainModel = 'models/5_1_encoder_decoder_warp0.1_relu_50000.pkl'
predictModel = 'models/'+exp_prefix+'predict'
imgoutdir = 'resimg'
Lr = 0.01 
batch = 32
trainstep = 50000
showiter = 100
snapshot = 10000

stateEncoderDecoder = StateEncoderDecoder()
predictNet = PredictNet()
# encode the input using pretrained model
print 'load pretrained...'
stateEncoderDecoder=loadPretrain(stateEncoderDecoder,preTrainModel)
stateEncoderDecoder.cuda()
predictNet.cuda()

criterion = nn.MSELoss()
# predictOptimizer = optim.SGD(predictNet.parameters(), lr = Lr, momentum=0.9)
predictOptimizer = optim.Adam(predictNet.parameters(), lr = Lr)

imgdataset = MnistManiDataset(fix_warping=True, warp_scale=0.3)
dataloader = DataLoader(imgdataset, batch_size=batch, shuffle=True, num_workers=4)

lossplot = []
running_loss = 0.0

ind = 0
while True:
	for sample in dataloader:  # loop over the dataset multiple times
		ind = ind + 1
		inputImgs = sample['data']
		outputImgs = sample['wdata']
		actions = sample['warpvec']
		# print inputImgs.size(), outputImgs.size(),actions.size()

		# print 'imput img shape:', inputImg.shape
		inputState = Variable(inputImgs.unsqueeze(1),volatile=True)
		expectPredict = Variable(outputImgs.unsqueeze(1),volatile=True)

		inputState = inputState.cuda()
		expectPredict = expectPredict.cuda()

		# zero the parameter gradients
		# stateOptimizer.zero_grad()
		predictOptimizer.zero_grad()

		# import ipdb; ipdb.set_trace()

		# forward + backward + optimize
		_, feature = stateEncoderDecoder(inputState)
		_, featureExpect = stateEncoderDecoder(expectPredict)
		featureExpect = Variable(featureExpect.data.squeeze().clone())

		# print feature.data[:,:,0,0].size(), torch.from_numpy(action).size()
		inputPredict = torch.cat((feature.data[:,:,0,0], actions.cuda()),dim=1)
		# print inputPredict.size()
		inputPredict = Variable(inputPredict,requires_grad=True)
		predict = predictNet(inputPredict)
		# print predict.size(),featureExpect.size()
		loss = criterion(predict, featureExpect)

		running_loss += loss.data[0]
		if ind % showiter == 0:    # print every 20 mini-batches
			print('[%d] loss: %.5f lr: %f' %
			(ind , running_loss / showiter, predictOptimizer.param_groups[0]['lr']))
			running_loss = 0.0
		lossplot.append(loss.data[0])

		loss.backward()
		predictOptimizer.step()

		if (ind)%snapshot==0:
			torch.save(predictNet.state_dict(), predictModel+'_'+str(ind)+'.pkl')

		if (ind)%30000==0 or (ind)%40000==0:
			Lr = Lr * 0.2
			for param_group in predictOptimizer.param_groups:
				param_group['lr'] = Lr
		if ind==trainstep:
			break

	if ind==trainstep:
		break
	


import matplotlib.pyplot as plt
lossplot = np.array(lossplot)
lossplot = lossplot.reshape((-1,10))
lossplot = lossplot.mean(axis=1)
plt.plot(lossplot)
plt.grid()
plt.savefig(join(imgoutdir, predictModel.split('/')[-1]+'.png'))
plt.ylim([0,1])
plt.show()
import ipdb; ipdb.set_trace()

