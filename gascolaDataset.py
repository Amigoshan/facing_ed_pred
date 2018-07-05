import cv2
import numpy as np
import os
from os.path import isfile, join, isdir
from os import listdir
import xml.etree.ElementTree
from torch.utils.data import Dataset, DataLoader
from utils import im_scale_norm_pad, img_denormalize, seq_show, im_crop, im_hsv_augmentation, put_arrow #_proper
import pandas as pd

import matplotlib.pyplot as plt

class GascolaDataset(Dataset):

    def __init__(self, csv_file='/datadrive/gascola_facing/all_annotations.csv',
                        images_folder = '/datadrive/gascola_facing',
                        imgsize = 192, data_aug = False,
                        mean=[0,0,0],std=[1,1,1]):

        self.imgsize = imgsize
        self.aug = data_aug
        self.mean = mean
        self.std = std
        self.csv_file = pd.read_csv(csv_file)
        self.images_folder = images_folder

        print len(self.csv_file)

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        point = self.csv_file.iloc[idx]
        #print(point_info)
        img_name = point['img']
        direction_angle = point['angle']
        x1, y1, x2, y2 = point['x1'], point['y1'], point['x2'], point['y2']

        direction_angle_cos = np.sin(float(direction_angle))#np.cos(float(direction_angle))
        direction_angle_sin = np.cos(float(direction_angle))#np.sin(float(direction_angle))
        label = np.array([direction_angle_cos, direction_angle_sin], dtype=np.float32)
        img = cv2.imread(os.path.join(self.images_folder, img_name))
        img_shape = img.shape
        img = img[max(0, y1-10): min(img_shape[0], y2+10), max(0, x1-10): min(img_shape[1], x2+10)]
        if self.aug:
            img = im_hsv_augmentation(img)
            img = im_crop(img)

        outimg = im_scale_norm_pad(img, outsize=self.imgsize, mean=self.mean, std=self.std, down_reso=True)

        return {'img':outimg, 'label':label}

if __name__=='__main__':
    # test 
    np.set_printoptions(precision=4)
    gascolaDataset = GascolaDataset()
    print len(gascolaDataset)
    for k in range(100):
        img = gascolaDataset[k*100]['img']
        label = gascolaDataset[k*100]['label']
        print img.dtype, label
        print np.max(img), np.min(img), np.mean(img)
        print img.shape
        img = img_denormalize(img)
        img = put_arrow(img, label)
        cv2.imshow('img',img)
        cv2.waitKey(0)

    dataloader = DataLoader(gascolaDataset, batch_size=4, shuffle=True, num_workers=1)

    dataiter = iter(dataloader)

    # import ipdb;ipdb.set_trace()

    for sample in dataloader:
      print sample['label'], sample['img'].size()
    print seq_show(sample['img'].numpy())