import cv2
import numpy as np

from os.path import isfile, join, isdir
from os import listdir
import xml.etree.ElementTree
from torch.utils.data import Dataset, DataLoader
from utils import im_scale_norm_pad, img_denormalize, seq_show, im_crop, im_hsv_augmentation, put_arrow
import pandas as pd

import matplotlib.pyplot as plt

class KoperLabelDataset(Dataset):

    def __init__(self, datafile='/datadrive/Ko-PER/koperdata',
                        imgdir='/datadrive/Ko-PER',
                        imgsize = 192, data_aug = False,
                        mean=[0,0,0],std=[1,1,1]):

        self.imgsize = imgsize
        self.aug = data_aug
        self.mean = mean
        self.std = std

        with open(datafile, 'r') as f:
            lines = f.readlines()

        self.datalist = []
        lineInd = 0
        while lineInd < len(lines):
            line = lines[lineInd].strip()
            lineInd += 1
            if line.split('.')[-1] == 'bmp':
                imgfile = line
                while True:
                    line = lines[lineInd].strip()
                    lineInd += 1
                    if line=='':
                        break
                    linesplit = line.split(' ')
                    bbox = [int(linesplit[0]),int(linesplit[1]),int(linesplit[2]),int(linesplit[3])]
                    angle = float(linesplit[4])
                    self.datalist.append({'img':join(imgdir, imgfile),'bbox':bbox, 'angle':angle})


        print len(self.datalist)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        img_name = self.datalist[idx]['img']
        bbox = direction_angle = self.datalist[idx]['bbox']
        direction_angle = self.datalist[idx]['angle']
        direction_angle = -direction_angle
        # print direction_angle
        direction_angle_cos = np.cos(float(direction_angle))
        direction_angle_sin = np.sin(float(direction_angle))
        label = np.array([direction_angle_sin, direction_angle_cos], dtype=np.float32)
        img = cv2.imread(img_name)
        img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        if self.aug:
            img = im_hsv_augmentation(img)
            img = im_crop(img)

        outimg = im_scale_norm_pad(img, outsize=self.imgsize, mean=self.mean, std=self.std, down_reso=True)

        return {'img':outimg, 'label':label}

if __name__=='__main__':
    # test 
    np.set_printoptions(precision=4)
    trackingLabelDataset = KoperLabelDataset(data_aug = True)
    print len(trackingLabelDataset)
    for k in range(100):
        img = trackingLabelDataset[k*100]['img']
        label = trackingLabelDataset[k*100]['label']
        print img.dtype, label
        print np.max(img), np.min(img), np.mean(img)
        print img.shape
        img = img_denormalize(img)
        img = put_arrow(img, label)
        cv2.imshow('img',img)
        cv2.waitKey(0)

    dataloader = DataLoader(trackingLabelDataset, batch_size=4, shuffle=True, num_workers=1)

    dataiter = iter(dataloader)

    # # import ipdb;ipdb.set_trace()

    for sample in dataloader:
      print sample['label'], sample['img'].size()
      print seq_show(sample['img'].numpy())
