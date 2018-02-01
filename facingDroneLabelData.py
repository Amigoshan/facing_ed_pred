import cv2
import numpy as np

from os.path import isfile, join, isdir
from os import listdir
import xml.etree.ElementTree
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt


class FacingDroneLabelDataset(Dataset):

    def __init__(self, imgdir='/home/wenshan/datasets/droneData/label',imgsize = 192):

        self.imgsize = imgsize
        self.imgnamelist = []
        self.labellist = []
        self.dir2ind = {'n': 0,'ne': 1,'e': 2, 'se': 3,'s': 4,'sw': 5,'w': 6,'nw': 7}

        imgind = 0
        for clsfolder in listdir(imgdir):
            
            clsind = self.dir2ind[clsfolder]

            clsfolderpath = join(imgdir, clsfolder)

            for imgname in listdir(clsfolderpath):
                if imgname[-3:] == 'jpg':
                    self.imgnamelist.append(join(clsfolderpath, imgname))
                    self.labellist.append(clsind)

        self.N = len(self.imgnamelist)
        print 'Read', self.N, 'images...'

    def img_normalize(self, img):
        img = img[:,:,[2,1,0]] # bgr to rgb
        img = img.astype(np.float32)/255.0
        img = img.transpose(2,0,1)
        return img

    def img_denormalize(self, img):
        print img.shape
        img = img.transpose(1,2,0)
        img = img.clip(0,1) # network can output values out of range
        img = (img*255).astype(np.uint8)
        img = img[:,:,[2,1,0]]
        return img

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        img = cv2.imread(self.imgnamelist[idx]) # in bgr
        label = self.labellist[idx]

        resize_scale = float(self.imgsize)/np.max(img.shape)
        img = cv2.resize(img, (0,0), fx = resize_scale, fy = resize_scale)
        img = self.img_normalize(img)
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
    for k in range(100):
        img = facingDroneLabelDataset[k]['img']
        print img.dtype, facingDroneLabelDataset[k]['label']
        print np.max(img), np.min(img), np.mean(img)
        print img.shape
        cv2.imshow('img',facingDroneLabelDataset.img_denormalize(img))
        cv2.waitKey(0)
