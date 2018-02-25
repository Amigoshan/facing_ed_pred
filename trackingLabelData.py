import cv2
import numpy as np

from os.path import isfile, join, isdir
from os import listdir
import xml.etree.ElementTree
from torch.utils.data import Dataset, DataLoader
from utils import im_scale_norm_pad, img_denormalize, seq_show, im_crop, im_hsv_augmentation
import pandas as pd

import matplotlib.pyplot as plt

class TrackingLabelDataset(Dataset):

    def __init__(self, csv_file='/datadrive/data/aayush/combined_data2/train/annotations/person_annotations.csv',imgsize = 192, data_aug = False):

        self.imgsize = imgsize
        self.aug = data_aug
        self.csv_file = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        point_info = self.csv_file.iloc[idx]
        #print(point_info)
        img_name = point_info['path']
        direction_angle = point_info['direction_angle']

        direction_angle_cos = np.cos(float(direction_angle))
        direction_angle_sin = np.sin(float(direction_angle))
        label = np.array([direction_angle_sin, direction_angle_cos], dtype=np.float32)
        img = cv2.imread(img_name)

        if self.aug:
            img = im_hsv_augmentation(img)
            img = im_crop(img)

        outimg = im_scale_norm_pad(img, outsize=192, down_reso=True)

        return {'img':outimg, 'label':label}

if __name__=='__main__':
    # test 
    np.set_printoptions(precision=4)
    trackingLabelDataset = TrackingLabelDataset()
    print len(trackingLabelDataset)
    for k in range(1):
        img = trackingLabelDataset[k*10]['img']
        print img.dtype, trackingLabelDataset[k*10]['label']
        print np.max(img), np.min(img), np.mean(img)
        print img.shape
        cv2.imshow('img',img_denormalize(img))
        cv2.waitKey(0)

    dataloader = DataLoader(trackingLabelDataset, batch_size=4, shuffle=True, num_workers=1)

    dataiter = iter(dataloader)

    # import ipdb;ipdb.set_trace()

    for sample in dataloader:
      print sample['label'], sample['img'].size()
      print seq_show(sample['img'].numpy())
