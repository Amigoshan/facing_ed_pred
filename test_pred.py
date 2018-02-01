import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from utils import loadPretrain2
from mnistManipulationData import MnistManiDataset
from StateEncoderDecoder import StateEncoderDecoder, StateEncoder, StateDecoder
from Predictor import PredictNet

import matplotlib.pyplot as plt

predictModel = 'models/7_3__predict_joint_50000.pkl'
encoderModel = 'models/7_3__encoder_joint_50000.pkl'
decoderModel = 'models/7_3__decoder_joint_50000.pkl'
# predictModel = 'models/2_2_predict_32_50000.pkl'
# edsave = 'models/1_1_encoder_decoder_50000.pkl'

stateEncoder = StateEncoder()
stateDecoder = StateDecoder()
predictNet = PredictNet([32,32,32])
# encode the input using pretrained model
print 'load pretrained...'
stateEncoder=loadPretrain2(stateEncoder,encoderModel)
stateDecoder=loadPretrain2(stateDecoder,decoderModel)
predictNet=loadPretrain2(predictNet,predictModel)
stateEncoder.cuda()
stateDecoder.cuda()
predictNet.cuda()

imgdataset = MnistManiDataset(cat = 'test', fix_warping=True, warp_scale=0.3)
dataloader = DataLoader(imgdataset, batch_size=1, shuffle=True, num_workers=4)

fig = plt.figure()
ax_img1 = fig.add_subplot(221)
ax_img1.set_title('Sample image on the map')
ax_img2 = fig.add_subplot(222)
ax_img2.set_title('Input image after action')
ax_pred1 = fig.add_subplot(223)
ax_pred1.set_title('Encode image on the map')
ax_pred2 = fig.add_subplot(224)
ax_pred2.set_title('Predict encode after action')

# for k in range(10):  # loop over the dataset multiple times
criterion = nn.MSELoss()

for ind,sample in enumerate(dataloader):  # loop over the dataset multiple times

	inputImg = sample['data']
	outputImg = sample['wdata']
	action = sample['warpvec']
	print 'action:',action
	# print inputImg.size(), outputImg.size(),action.size()

	inputState = Variable(inputImg.unsqueeze(1).cuda(),volatile=True)
	encodeState = stateEncoder(inputState) # encode state before action 
	inputPredict = torch.cat((encodeState.data[:,:,0,0], action.cuda()),dim=1)
	inputPredict = Variable(inputPredict,volatile=True)
	predictState = predictNet(inputPredict) # encode of predicted state 

	# print the encoder output to the the magnitude
	eninput = encodeState.data.cpu().numpy().squeeze()
	print 'Encode state before action:', eninput
	print 'Mean:', np.mean(eninput), 'Std:', np.std(eninput)

	featureExpect = stateEncoder(Variable(outputImg.unsqueeze(1).cuda(),volatile=True)) # encode state after action 
	# featureExpect = Variable(featureExpect.data[:,:,0,0]) # for calculating the predict loss
	# print 'shape:',predictState.size(),featureExpect.size()
	loss = criterion(predictState,featureExpect[:,:,0,0])
	print 'loss:', loss.data[0]
	# print 'predict:',predictState.data.cpu().numpy()
	# print 'expect:',featureExpect[:,:,0,0].data.cpu().numpy()

	predictState = predictState.unsqueeze(-1).unsqueeze(-1)
	predictDecode = stateDecoder(predictState)

	outputState = stateEncoder(Variable(outputImg.unsqueeze(1).cuda(),volatile=True))
	outputDecode = stateDecoder(outputState)

	# visualize the predict and goundtruth
	imgdataset.showdigit(np.array([inputImg.squeeze().numpy(),outputImg.squeeze().numpy(),
		outputDecode.data.squeeze().cpu().numpy(),predictDecode.data.squeeze().cpu().numpy()]))

	# inputImg = mapdataset.mapDeNormalize(inputImg.numpy()[0,:,:,:]) #1
	# ax_img1.imshow(inputImg)
	# outputPlot = mapdataset.mapDeNormalize(outputImg.numpy()[0,:,:,:]) #2
	# ax_img2.imshow(outputPlot)

	# outputState = stateEncoder(Variable(outputImg.cuda(),volatile=True))
	# outputDecode = stateDecoder(outputState)
	# outputDecode = outputDecode.data[0,:,:,:].cpu().numpy()
	# outputDecode = mapdataset.mapDeNormalize(outputDecode) #3
	# ax_pred1.imshow(outputDecode)

	# predictDecode = predictDecode.data[0,:,:,:].cpu().numpy()
	# predictImg = mapdataset.mapDeNormalize(predictDecode)
	# ax_pred2.imshow(predictImg)

	# plt.show(block=False)
	# temp=raw_input()
	# if len(temp)>0 and temp[0]=='q':
	# 	break

