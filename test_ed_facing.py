import cv2
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from StateEncoderDecoder import StateEncoderDecoder
from facingData import FacingDataset
from facingLabelData import FacingLabelDataset
from utils import loadPretrain

import matplotlib.pyplot as plt
import numpy as np
import math
from os.path import join
np.set_printoptions(threshold=np.nan, precision=2, suppress=True)


testnum = 680
encodename = '1_2_encoder_decoder_facing_leaky_50000'
encodedecoderSave = 'models_facing/'+encodename+'.pkl'

imgoutdir = 'resimg_facing'

hiddens = [3,16,32,32,64,64,128,256] 
kernels = [4,4,4,4,4,4,3]
paddings = [1,1,1,1,1,1,0]
strides = [2,2,2,2,2,2,1]


stateEncoderDecoder = StateEncoderDecoder(hiddens, kernels, strides, paddings, actfunc='leaky')
# encode the input using pretrained model
print 'load pretrained...'
stateEncoderDecoder=loadPretrain(stateEncoderDecoder,encodedecoderSave)
stateEncoderDecoder.cuda()

# testing
# imgdataset = FacingDataset()
imgdataset = FacingLabelDataset()
dataloader = DataLoader(imgdataset, batch_size=1, shuffle=True, num_workers=1)

criterion = nn.MSELoss()

codelist = []
labellist = []
codemeanlist, codestdlist, codezerolist = [], [], []
for ind,sample in enumerate(dataloader):  # loop over the dataset multiple times

    # img = sample
    img = sample['img']
    label = sample['label']
    inputState = Variable(img.cuda(),volatile=True)
    targetState = Variable(img.clone().cuda())
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
    codemean, codestd, codezero = np.mean(eninput), np.std(eninput), np.sum(np.abs(eninput)<1e-5)
    print 'Mean:', codemean, 'Std:', codestd, '#0:', codezero
    codemeanlist.append(codemean)
    codestdlist.append(codestd)
    codezerolist.append(codezero)

    diff = output.data.cpu() - targetState.data.cpu()
    diffimg = (np.absolute(diff.numpy()[0,0,:,:]))
    # print np.max(diffimg),np.mean(diffimg)

    print 'loss:',loss.data[0]
    codelist.append(code.data.cpu().squeeze().numpy())
    labellist.append(label[0])

    # print 'output size',output.size()
    # img_input = img.squeeze().numpy()
    # img_input = imgdataset.img_denormalize(img_input)
    # img_show = output.data.cpu().squeeze().numpy()
    # img_show = imgdataset.img_denormalize(img_show)
    # img_show = np.concatenate((img_input,img_show),axis=1)
    # cv2.imshow('img',img_show)
    # cv2.waitKey(0)


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

plt.legend(ptslist, [str(x) for x in range(8)])
plt.savefig(join(imgoutdir, encodename+'_test.png'))
plt.show()
