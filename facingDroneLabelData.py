import cv2
import numpy as np

from os.path import isfile, join, isdir
from os import listdir
import xml.etree.ElementTree
from torch.utils.data import Dataset, DataLoader
from utils import img_normalize, img_denormalize, seq_show

import matplotlib.pyplot as plt


class FacingDroneLabelDataset(Dataset):

    def __init__(self, imgdir='/datasets/droneData/label',imgsize = 192):

        self.imgsize = imgsize
        self.imgnamelist = []
        self.labellist = []
        # self.dir2ind = {'n': 0,'ne': 1,'e': 2, 'se': 3,'s': 4,'sw': 5,'w': 6,'nw': 7}
        self.dir2val = {'n':  [1., 0.],
                        'ne': [0.707, 0.707],
                        'e':  [0., 1.],
                        'se': [-0.707, 0.707],
                        's':  [-1., 0.],
                        'sw': [-0.707, -0.707],
                        'w':  [0., -1.],
                        'nw': [0.707, -0.707]}


        imgind = 0
        for clsfolder in listdir(imgdir):
            
            clsval = self.dir2val[clsfolder]

            clsfolderpath = join(imgdir, clsfolder)

            for imgname in listdir(clsfolderpath):
                if imgname[-3:] == 'jpg':
                    self.imgnamelist.append(join(clsfolderpath, imgname))
                    self.labellist.append(clsval)

        self.N = len(self.imgnamelist)
        print 'Read', self.N, 'images...'

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        img = cv2.imread(self.imgnamelist[idx]) # in bgr
        label = np.array(self.labellist[idx], dtype=np.float32)

        resize_scale = float(self.imgsize)/np.max(img.shape)
        img = cv2.resize(img, (0,0), fx = resize_scale, fy = resize_scale)
        img = img_normalize(img)
        # print img.shape
        imgw = img.shape[2]
        imgh = img.shape[1]
        startx = (self.imgsize-imgw)/2
        starty = (self.imgsize-imgh)/2
        # print startx, starty
        outimg = np.zeros((3,self.imgsize,self.imgsize), dtype=np.float32)
        outimg[:, starty:starty+imgh, startx:startx+imgw] = img

        return {'img':outimg, 'label':label}

if __name__=='__main__':
    # test 
    np.set_printoptions(precision=4)
    facingDroneLabelDataset = FacingDroneLabelDataset()
    for k in range(1):
        img = facingDroneLabelDataset[k*10]['img']
        print img.dtype, facingDroneLabelDataset[k*10]['label']
        print np.max(img), np.min(img), np.mean(img)
        print img.shape
        cv2.imshow('img',img_denormalize(img))
        cv2.waitKey(0)

    dataloader = DataLoader(facingDroneLabelDataset, batch_size=4, shuffle=True, num_workers=1)

    for sample in dataloader:
      print sample['label'], sample['img'].size()
      print seq_show(sample['img'].numpy())
