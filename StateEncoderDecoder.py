import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable

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


class EncoderReg(nn.Module):

    def __init__(self, hiddens=None, kernels=None, strides=None, paddings=None, actfunc='relu', regnum=2):
        super(EncoderReg,self).__init__()
        self.encoder = StateEncoder(hiddens, kernels, strides, paddings, actfunc)
        self.reg = nn.Linear(hiddens[-1], regnum)

    def forward(self,x):
        x_encode = self.encoder(x)
        x = self.reg(x_encode.view(x_encode.size()[0], -1))
        return x, x_encode

class EncoderReg_norm(nn.Module):

    def __init__(self, hiddens=None, kernels=None, strides=None, paddings=None, actfunc='relu', regnum=2):
        super(EncoderReg_norm,self).__init__()
        self.encoder = StateEncoder(hiddens, kernels, strides, paddings, actfunc)
        self.reg = nn.Linear(hiddens[-1], regnum)

    def forward(self,x):
        x_encode = self.encoder(x)
        x = self.reg(x_encode.view(x_encode.size()[0], -1))
        y = x.abs() # normalize so |x| + |y| = 1
        y = y.sum(dim=1) 
        # import ipdb; ipdb.set_trace()
        x = x/y.unsqueeze(1)
        return x, x_encode

class EncoderReg_Pred(nn.Module):

    def __init__(self, hiddens=None, kernels=None, strides=None, paddings=None, actfunc='relu', regnum=2, rnnHidNum=128):
        super(EncoderReg_Pred,self).__init__()
        self.encoder = StateEncoder(hiddens, kernels, strides, paddings, actfunc)
        self.reg = nn.Linear(hiddens[-1], regnum)
        self.codenum = hiddens[-1]

        self.rnnHidNum = rnnHidNum
        self.pred_en = nn.LSTM(hiddens[-1], rnnHidNum)
        self.pred_de = nn.LSTM(hiddens[-1], rnnHidNum)
        self.pred_de_linear = nn.Linear(self.rnnHidNum, self.codenum)

    def init_hidden(self, hidden_size, batch_size):
        return (Variable(torch.zeros(1, batch_size, hidden_size)).cuda(),
                Variable(torch.zeros(1, batch_size, hidden_size)).cuda())

    def forward(self,x):
        x_encode = self.encoder(x)
        batchsize = x_encode.size()[0]
        x_encode = x_encode.view(batchsize, -1)
        # regression the direction
        x = self.reg(x_encode)
        # y = x.abs() # normalize so |x| + |y| = 1
        # y = y.sum(dim=1) 
        # x = x/y.unsqueeze(1)

        # rnn predictor
        innum = batchsize/2 # use first half as input, last half as target
        pred_in = x_encode[0:innum,:].unsqueeze(1) # input of LSTM should be T x batch x InLen
        hidden = self.init_hidden(self.rnnHidNum, 1)
        pred_en_out, hidden = self.pred_en(pred_in, hidden)

        # import ipdb; ipdb.set_trace()
        pred_de_in = Variable(torch.zeros(1,1,self.codenum)).cuda()
        pred_out = []
        for k in range(innum, batchsize): # input the decoder one by one cause there's a loop
            pred_de_out, hidden = self.pred_de(pred_de_in, hidden)
            pred_de_out = self.pred_de_linear(pred_de_out.view(1, self.rnnHidNum))
            pred_out.append(pred_de_out)
            pred_de_in = pred_de_out.detach().unsqueeze(1)

        pred_out = torch.cat(tuple(pred_out), dim=0)
        return x, x_encode, pred_out

if __name__ == '__main__':
    

    hiddens = [3,16,32,32,64,64,128,256] 
    kernels = [4,4,4,4,4,4,3]
    paddings = [1,1,1,1,1,1,0]
    strides = [2,2,2,2,2,2,1]

    datasetdir='/home/wenshan/datasets'
    unlabel_batch = 4
    lr = 0.005

    from facingDroneUnlabelData import FacingDroneUnlabelDataset
    from torch.utils.data import DataLoader
    from os.path import join
    import torch.nn as nn
    import torch.optim as optim

    stateEncoder = EncoderReg_Pred(hiddens, kernels, strides, paddings, actfunc='leaky',rnnHidNum=128)
    print stateEncoder
    paramlist = list(stateEncoder.parameters())
    # for par in paramlist:
    #     print par.size()
    print len(paramlist)
    stateEncoder.cuda()
    imgdataset = FacingDroneUnlabelDataset(imgdir=join(datasetdir,'dirimg'), 
                                       batch = unlabel_batch, data_aug=True, extend=False)    
    dataloader = DataLoader(imgdataset, batch_size=1, shuffle=True, num_workers=1)

    criterion = nn.MSELoss()
    regOptimizer = optim.SGD(stateEncoder.parameters(), lr = lr, momentum=0.9)
    # regOptimizer = optim.Adam(stateEncoder.parameters(), lr = lr)

    lossplot = []
    encodesumplot = []
    ind = 0
    for sample in dataloader:
        ind += 1
        inputVar = Variable(sample.squeeze()).cuda()
        # print inputVar.size()
        x, encode, pred = stateEncoder(inputVar)
        # print encode.size(), x.size(), pred.size()

        # print encode

        pred_target = encode[unlabel_batch/2:,:].detach()

        loss_pred = criterion(pred, pred_target)

        # # loss = loss_label + loss_pred * lamb #+ normloss * lamb2

        # # zero the parameter gradients
        regOptimizer.zero_grad()
        loss_pred.backward()
        # # loss.backward()
        regOptimizer.step()

        lossplot.append(loss_pred.data[0])
        encodesumplot.append(encode.mean().data[0])
        print ind,loss_pred.data[0], encode.mean().data[0]

        if ind>=1000:
            break


    import matplotlib.pyplot as plt
    plt.plot(lossplot)
    plt.plot(encodesumplot)
    plt.grid()
    plt.show()
