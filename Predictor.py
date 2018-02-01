import torch
import torch.nn as nn
import torch.nn.functional as F

hiddens_g = [32,64,32]

class PredictNet(nn.Module):

	def __init__(self, hiddens = None, actionNum = 8):
		super(PredictNet,self).__init__()
		global hiddens_g
		if not hiddens:
			hiddens = hiddens_g
		self.predict = nn.Sequential()
		self.predict.add_module('fc1', nn.Linear(hiddens[0]+actionNum, hiddens[1]))
		self.predict.add_module('relu1' , nn.LeakyReLU(0.1,inplace=True))
		for k in range(1,len(hiddens)-1):
			fc = nn.Linear(hiddens[k], hiddens[k+1])
			self.predict.add_module('fc%d' % (k + 1), fc)
			if k<len(hiddens)-2:
				self.predict.add_module('relu%d'  % (k + 1), nn.LeakyReLU(0.1,inplace=True))
			
		self._initialize_weights()

	def forward(self, x):
		
		return self.predict(x)

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

# net = PredictNet([10,30,50,100,20])
# print net