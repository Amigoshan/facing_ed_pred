import cv2
import numpy as np

from os.path import isfile, join, isdir
from os import listdir
import xml.etree.ElementTree
from torch.utils.data import Dataset, DataLoader
from utils import im_scale_norm_pad, img_denormalize, seq_show, im_crop, im_hsv_augmentation

import matplotlib.pyplot as plt


class FacingDroneLabelDataset(Dataset):

    def __init__(self, imgdir='/home/wenshan/datasets/droneData/label',imgsize = 192, data_aug = False):

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
        self.aug = data_aug

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

        if self.aug:
            img = im_hsv_augmentation(img)
            img = im_crop(img)


        outimg = im_scale_norm_pad(img, outsize=192, down_reso=True)

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

    dataloader = DataLoader(facingDroneLabelDataset, batch_size=40, shuffle=True, num_workers=1)

    dataiter = iter(dataloader)

    import ipdb;ipdb.set_trace()

    for sample in dataloader:
      print sample['label'], sample['img'].size()
      print seq_show(sample['img'].numpy())

