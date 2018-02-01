import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# these are default values of the network
# if the initial values are not assigned, these values will be used
hiddens_g = [1,8,16,16,32,32] # 14, 7, 4, 2, 1
kernels_g = [4,4,3,4,2]
paddings_g = [1,1,1,1,0]
strides_g = [2,2,2,2,1]

class StateEncoder(nn.Module):

	def __init__(self, hiddens=None, kernels=None, strides=None, paddings=None, actfunc='relu'):
		super(StateEncoder,self).__init__()
		global hiddens_g,kernels_g,paddings_g,strides_g
		if not hiddens:
			hiddens = hiddens_g
		if not kernels:
			kernels = kernels_g
		if not strides:
			strides = strides_g
		if not paddings:
			paddings = paddings_g

		self.encoder = nn.Sequential()
		for k in range(len(hiddens)-1):
			conv = nn.Conv2d(hiddens[k], hiddens[k+1],kernels[k],
								stride=strides[k],padding=paddings[k])
			self.encoder.add_module('conv%d' % (k + 1), conv)
			# if k<len(hiddens)-2:
			if actfunc=='leaky':
				self.encoder.add_module('relu%d'  % (k + 1), nn.LeakyReLU(0.1,inplace=True))
			else:
				self.encoder.add_module('relu%d'  % (k + 1), nn.ReLU(inplace=True))
			
		self._initialize_weights()

	def forward(self, x):

		return self.encoder(x)

	def _initialize_weights(self):
		for m in self.modules():
			# print type(m)
			if isinstance(m, nn.Conv2d):
				# print 'conv2d'
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				# print 'batchnorm'
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				# print 'linear'
				n = m.weight.size(1)
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()

class StateDecoder(nn.Module):

	def __init__(self, hiddens=None, kernels=None, strides=None, paddings=None, actfunc='relu'):
		super(StateDecoder,self).__init__()
		global hiddens_g,kernels_g,paddings_g,strides_g
		if not hiddens:
			hiddens = hiddens_g[::-1]
		if not kernels:
			kernels = kernels_g[::-1]
		if not strides:
			strides = strides_g[::-1]
		if not paddings:
			paddings = paddings_g[::-1]

		self.decoder = nn.Sequential()
		for k in range(len(hiddens)-1):
			conv = nn.ConvTranspose2d(hiddens[k], hiddens[k+1],kernels[k],
								stride=strides[k],padding=paddings[k])
			self.decoder.add_module('deconv%d' % (k + 1), conv)
			if actfunc=='leaky':
				self.decoder.add_module('relu%d'  % (k + 1), nn.LeakyReLU(0.1,inplace=True))
			else:
				self.decoder.add_module('relu%d'  % (k + 1), nn.ReLU(inplace=True))

		self._initialize_weights()

	def forward(self, x):

		return self.decoder(x)

	def _initialize_weights(self):
		for m in self.modules():
			# print type(m)
			if isinstance(m, nn.Conv2d):
				# print 'conv2d'
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				# print 'batchnorm'
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				# print 'linear'
				n = m.weight.size(1)
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()


class StateEncoderDecoder(nn.Module):

	def __init__(self, hiddens=None, kernels=None, strides=None, paddings=None, actfunc='relu'):
		super(StateEncoderDecoder,self).__init__()
		self.encoder = StateEncoder(hiddens, kernels, strides, paddings, actfunc)
		if hiddens: 
			hiddens = hiddens[::-1]
		if kernels:
			kernels = kernels[::-1]
		if strides:
			strides = strides[::-1]
		if paddings:
			paddings = paddings[::-1]
		self.decoder = StateDecoder(hiddens, kernels, strides, paddings, actfunc)

	def forward(self,x):
		x_encode = self.encoder(x)
		x = self.decoder(x_encode)
		return x, x_encode

	def forward_encoder(self,x):
		x = self.encoder(x)
		return x

	def forward_decoder(self,x):
		x = self.decoder(x)
		return x


class EncoderCls(nn.Module):

	def __init__(self, hiddens=None, kernels=None, strides=None, paddings=None, actfunc='relu', clsnum=8):
		super(EncoderCls,self).__init__()
		self.encoder = StateEncoder(hiddens, kernels, strides, paddings, actfunc)
		self.cls = nn.Linear(hiddens[-1], clsnum)

	def forward(self,x):
		x_encode = self.encoder(x)
		x = self.cls(x_encode.view(x_encode.size()[0], -1))
		return x, x_encode

# hiddens = [3,16,32,64,128,256]
# kernels = [5,5,5,5,5]
# paddings = [2,2,2,2,0]
# strides = [2,2,2,2,1]

# stateEncoder = StateEncoder()
# print stateEncoder
# from PacmanDataset import PacmanDataset
# pacmandataset = PacmanDataset()
# from torch.autograd import Variable
# inputVar = Variable(torch.from_numpy(pacmandataset[0]['img'])).unsqueeze(0)
# print inputVar
# print inputVar.size()
# encode = stateEncoder(inputVar)
# print encode.size()

# stateDecoder = StateDecoder()
# print stateDecoder
# decode = stateDecoder(encode)
# print decode.size()