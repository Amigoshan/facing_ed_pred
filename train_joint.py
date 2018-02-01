import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from os.path import join

from utils import loadPretrain2
from mnistManipulationData import MnistManiDataset
from StateEncoderDecoder import StateEncoderDecoder, StateEncoder, StateDecoder
from Predictor import PredictNet



expInd = '7_3_'
LoadPretrain = False

lr_pred = 0.001
lr_state = 0.001
lr_state_when_pred = 0.001
batch = 64
showiter = 10
trainstep = 10000
snapshot = 5000

pred_num = 7
ed_num = 3

lamb = 0.5

predhidden = [32,32,32]

def updateTrainState(step):
	global trainState
	global trainPred
	# trainState = True
	# trainPred = False
	if (step-ed_num)%(pred_num+ed_num) == 0:
		trainState = False
		trainPred = True
	elif (step)%(pred_num+ed_num) == 0:
		trainState = True
		trainPred = False
	else:
		pass


startind = 4
for predhidden in [[32,32,32],[32,16,32]]:
	for lamb in [0.1,0.2,0.3,0.4,0.5,0.6,0.7]:

		expInd = '7_' +str(startind)+ '_'
		startind += 1

		# edModel = 'models/encoder_decoder_param_conv5_3.pkl'
		# predictModel = 'models/predict_fc3_512_act_lre-3_batch50.pkl'
		edModel = 'models/5_1_encoder_decoder_warp0.1_relu_50000.pkl'
		predModel = 'models/6_1_predict_50000.pkl'
		outdir = 'resimg'

		exp_name = 'joint'+ '_' +str(predhidden[1])+ '_lamb'+str(lamb)
		encodeSave = 'models/'+expInd+'encoder_'+exp_name
		decodeSave = 'models/'+expInd+'decoder_'+exp_name
		predictSave = 'models/'+expInd+'predict_'+exp_name


		stateEncoder = StateEncoder()
		stateDecoder = StateDecoder()
		predictNet = PredictNet(predhidden)
		if LoadPretrain:
			# encode the input using pretrained model
			print 'load pretrained...'
			stateEncoder=loadPretrain2(stateEncoder,edModel)
			stateDecoder=loadPretrain2(stateDecoder,edModel)
			predictNet=loadPretrain2(predictNet,predModel)
		stateEncoder.cuda()
		stateDecoder.cuda()
		predictNet.cuda()

		stateCriterion = nn.MSELoss()
		predictCriterion = nn.MSELoss()
		# encoderOptimizer = optim.SGD(stateEncoder.parameters(), lr = lr_state, momentum=0)
		# decoderOptimizer = optim.SGD(stateDecoder.parameters(), lr = lr_state, momentum=0.9)
		# predictOptimizer = optim.SGD(predictNet.parameters(), lr = lr_pred, momentum=0.9)
		encoderOptimizer = optim.Adam(stateEncoder.parameters(), lr = lr_state)
		decoderOptimizer = optim.Adam(stateDecoder.parameters(), lr = lr_state)
		predictOptimizer = optim.Adam(predictNet.parameters(), lr = lr_pred)
		encoder4predOptimizer = optim.Adam(stateEncoder.parameters(), lr = lr_state_when_pred)

		# data loader 
		imgdataset = MnistManiDataset(fix_warping=True, warp_scale=0.3)
		dataloader = DataLoader(imgdataset, batch_size=batch, shuffle=True, num_workers=4)

		lossplot_ed = []
		lossplot_pred = []
		running_loss_ed = 0.0
		running_loss_predict = 0.0

		trainState = True
		trainPred = False

		ind = 0
		while True:

			for sample in dataloader:  # loop over the dataset multiple times
				ind = ind + 1

				inputImgs = sample['data'].unsqueeze(1)
				outputImgs = sample['wdata'].unsqueeze(1)
				actions = sample['warpvec']
				# print inputImgs.size(), outputImgs.size(),actions.size()

				# print 'imput img shape:', inputImg.shape
				inputState = Variable(inputImgs.cuda(),requires_grad=True)
				expectPredict = Variable(outputImgs.cuda(),volatile=True)
				decodeTarget = Variable(inputImgs.clone().cuda(),requires_grad=False) # ed target

				# zero the parameter gradients
				encoderOptimizer.zero_grad()
				decoderOptimizer.zero_grad()
				predictOptimizer.zero_grad()
				encoder4predOptimizer.zero_grad()

				# forward + backward + optimize
				feature = stateEncoder(inputState)

				featureExpect = stateEncoder(expectPredict)
				featureExpect = Variable(featureExpect.data[:,:,0,0].clone()) # for calculating the predict loss

				decodeState = stateDecoder(feature) # for calculating the ed loss
				decodeloss = stateCriterion(decodeState, decodeTarget)

				inputPredict = torch.cat((feature[:,:,0,0], Variable(actions.cuda())),dim=1)
				predict = predictNet(inputPredict)
				# print predict.size(),featureExpect.size()
				predictloss = predictCriterion(predict, featureExpect)

				loss = lamb * decodeloss + (1-lamb) * predictloss
				loss.backward()
				decoderOptimizer.step()
				predictOptimizer.step()
				encoderOptimizer.step()

				# updateTrainState(ind)

				# if trainState:
				# 	# for param_group in encoderOptimizer.param_groups:
				# 	# 	param_group['lr'] = lr_state
				# 	decodeloss.backward()
				# 	decoderOptimizer.step()
				# 	encoderOptimizer.step()
				# 	trainStr = 'train ED '

				# if trainPred:
				# 	# for param_group in encoderOptimizer.param_groups:
				# 	# 	param_group['lr'] = lr_state_when_pred

				# 	predictloss.backward()
				# 	predictOptimizer.step()
				# 	# encoderOptimizer.step()
				# 	encoder4predOptimizer.step()
				# 	trainStr = 'train PRED '

				trainStr = ''

				running_loss_ed += decodeloss.data[0]
				running_loss_predict += predictloss.data[0]
				if ind % showiter == 0:    # print every 20 mini-batches
					timestr = time.strftime('%m/%d %H:%M:%S',time.localtime())
					print('[%d %s %s %s] ed-loss: %.5f pred-loss: %.5f lr: %f' %
					(ind, expInd, timestr, trainStr, running_loss_ed / showiter, running_loss_predict / showiter,
					predictOptimizer.param_groups[0]['lr']))
					running_loss_ed = 0.0
					running_loss_predict = 0.0
				lossplot_ed.append(decodeloss.data[0])
				lossplot_pred.append(predictloss.data[0])


				# if (ind)%50000==0:
				# 	Lr = Lr * 0.2
				# 	for param_group in predictOptimizer.param_groups:
				# 		param_group['lr'] = Lr

				if (ind)%snapshot==0:
					torch.save(predictNet.state_dict(), predictSave+ '_' + str(ind)+'.pkl')
					torch.save(stateEncoder.state_dict(), encodeSave+ '_' + str(ind)+'.pkl')
					torch.save(stateDecoder.state_dict(), decodeSave+ '_' + str(ind)+'.pkl')

				if ind==trainstep:
					break
			if ind==trainstep:
				break

		import matplotlib.pyplot as plt
		group = 10
		fig = plt.figure()
		ax1 = fig.add_subplot(121)
		if len(lossplot_ed)%group>0:
			lossplot_ed = lossplot_ed[0:len(lossplot_ed)/group*group]
		lossplot_ed = np.array(lossplot_ed)
		lossplot_ed = lossplot_ed.reshape((-1,group))
		lossplot_ed = lossplot_ed.mean(axis=1)
		ax1.plot(lossplot_ed)
		ax1.grid()

		ax2 = fig.add_subplot(122)
		if len(lossplot_pred)%group>0:
			lossplot_pred = lossplot_pred[0:len(lossplot_pred)/group*group]
		lossplot_pred = np.array(lossplot_pred)
		lossplot_pred = lossplot_pred.reshape((-1,group))
		lossplot_pred = lossplot_pred.mean(axis=1)
		ax2.plot(lossplot_pred)
		ax2.set_ylim([0,20])
		ax2.grid()
		plt.savefig(join(outdir,expInd + exp_name +'.png'), pad_inches=0)
		# plt.show()

import ipdb; ipdb.set_trace()


