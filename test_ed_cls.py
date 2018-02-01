import cv2
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from StateEncoderDecoder import StateEncoderDecoder
from mnistManipulationData import MnistManiDataset
from utils import loadPretrain

import matplotlib.pyplot as plt
import numpy as np
import math
from os.path import join
np.set_printoptions(threshold=np.nan, precision=2, suppress=True)


testnum = 1000
encodename = '5_2_encoder_decoder_warp0.1_relu_50000'
encodedecoderSave = 'models/'+encodename+'.pkl'

imgoutdir = 'resimg'

stateEncoderDecoder = StateEncoderDecoder()
# encode the input using pretrained model
print 'load pretrained...'
stateEncoderDecoder=loadPretrain(stateEncoderDecoder,encodedecoderSave)
stateEncoderDecoder.cuda()

# testing
mnistManiDataset = MnistManiDataset(cat = 'test') #, fix_warping=True, warp_scale=0.3)
dataloader = DataLoader(mnistManiDataset, batch_size=1, shuffle=True, num_workers=1)

criterion = nn.MSELoss()

codelist = []
labellist = []
codemeanlist, codestdlist, codezerolist = [], [], []
for ind,sample in enumerate(dataloader):  # loop over the dataset multiple times

    inputState = Variable(sample['wdata'].unsqueeze(1).cuda(),volatile=True)
    targetState = Variable(sample['wdata'].clone().unsqueeze(1).cuda())
    # print inputState.size()

    # forward 
    output, code = stateEncoderDecoder(inputState)
    loss = criterion(output, targetState)
    # print output.data.cpu().numpy(),np.max(output.data.cpu().numpy()),np.min(output.data.cpu().numpy())
    # print output.data.cpu().numpy()

    # some statistics on the coding
    eninput = code.data.cpu().numpy().squeeze()
    print '---'
    print 'Encode state:', eninput
    codemean, codestd, codezero = np.mean(eninput), np.std(eninput), np.sum(eninput<1e-5)
    print 'Mean:', codemean, 'Std:', codestd, '#0:', codezero
    codemeanlist.append(codemean)
    codestdlist.append(codestd)
    codezerolist.append(codezero)

    diff = output.data.cpu() - targetState.data.cpu()
    diffimg = (np.absolute(diff.numpy()[0,0,:,:]))
    # print np.max(diffimg),np.mean(diffimg)

    print 'loss:',loss.data[0]
    codelist.append(code.data.cpu().squeeze().numpy())
    labellist.append(sample['label'][0])

    inputimg = sample['wdata'].squeeze().numpy()
    outputimg = output.data.cpu().squeeze().numpy()
    # mnistManiDataset.showdigit(np.array([inputimg, outputimg, diffimg]), time=0)

    if ind==testnum-1:
        break

print '***'
print 'Mean:', np.mean(np.array(codemeanlist)), \
        'Std:', np.mean(np.array(codestdlist)), \
        '#0:', np.mean(np.array(codezerolist))

# import ipdb; ipdb.set_trace()
# convert the encoding results to 2d space
from sklearn.manifold import TSNE
model = TSNE(n_components=2)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', (1,0.647,0), (0.5,0.5,0.5), (0.5,0.7,1)]
# fig = plt.figure()

codes_tsne = model.fit_transform(np.array(codelist))
labellist = np.array(labellist)
# fc6_tsne = fig.add_subplot(131)
ptslist = []
for ind,color in enumerate(colors):
    codes_ind = labellist == ind
    points = plt.scatter(codes_tsne[codes_ind,0],codes_tsne[codes_ind,1],color=color, label=str(ind))
    ptslist.append(points)

plt.legend(ptslist, [str(x) for x in range(10)])
plt.savefig(join(imgoutdir, encodename+'_test.png'))
plt.show()
