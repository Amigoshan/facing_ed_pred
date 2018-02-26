import cv2
import numpy as np

from os.path import isfile, join, isdir
from os import listdir
import xml.etree.ElementTree
from torch.utils.data import Dataset, DataLoader
from utils import im_scale_norm_pad, img_denormalize, seq_show, im_hsv_augmentation, im_crop
import random
import matplotlib.pyplot as plt
import pandas as pd


class TrackingUnlabelDataset(Dataset):

    def __init__(self, imgdir='/datadrive/data/aayush/combined_data2/',
                        imgsize = 192, batch = 32, data_aug=False,
                        mean=[0,0,0],std=[1,1,1]):

        self.imgsize = imgsize
        self.imgnamelist = []
        # self.fileprefix = 'drone_'
        self.folderlist = ['train'] # can add val
        self.class_type = ['person']  # can add 'person'
        self.batch = batch
        self.aug = data_aug
        self.mean = mean
        self.std = std
        self.episodeNum = []


        for f_ind, foldername in enumerate(self.folderlist):

            folderpath = join(imgdir, foldername)
            # filenum = len(listdir(folderpath))
            # print 'find %d entities in %s ' % (filenum, folderpath) 
            
            for cls_type in self.class_type:
                filename = join(folderpath,'annotations','%s_annotations.csv' %cls_type)
                imglist = pd.read_csv(filename)['path']
                print(len(imglist))
                # path is of type:
                # /datadrive/data/aayush/combined_data2/train/images/ne/VIRAT_S_000207_00_000000_000045_764_158_ne_car_958_1071_575_658.png
                # filename_root, frame_num, obj_id, ...
                imglist_per_obj = {}
                for x in imglist:
                    obj_id = x.strip().split('_')[-7]
                    video_name = '_'.join(x.strip().split('/')[-1].split('_')[:6])
                    key = (video_name, obj_id)
                    if key not in imglist_per_obj:
                        imglist_per_obj[key] = []
                    imglist_per_obj[key].append(x)

                for k, v in imglist_per_obj.items():
                    ## need to sort according to frame_num
                    imglist = sorted(v, key=lambda x:x.strip().split('_')[-8])
                    # missimg = 0
                    lastind = -1
                    sequencelist = []
                    for filename in imglist:
                        if filename.split('.')[-1]!='png': # only process png file
                            continue
                        fileind = filename.strip().split('_')[-8]
                        try:
                            fileind = int(fileind)
                        except:
                            print 'filename parse error:', filename, fileind
                            continue
                        # filename = self.fileprefix+foldername+'_'+str(imgind)+'.jpg'
                        filepathname = filename

                        if lastind<0 or fileind==lastind+1:
                        # if isfile(filepathname):
                            sequencelist.append(filepathname)
                            lastind = fileind
                            # if missimg>0:
                                # print '  -- last missimg', missimg
                            # missimg = 0
                        else: # the index is not continuous
                            if len(sequencelist)>=batch:
                                # missimg = 1
                                self.imgnamelist.append(sequencelist)
                                # print 'image lost:', filename
                                print '** sequence: ', len(sequencelist)
                                # print sequencelist
                            sequencelist = []
                            lastind = -1
                            # else:
                                # missimg += 1
                    if len(sequencelist)>=batch:          
                        self.imgnamelist.append(sequencelist)
                        print '** sequence: ', len(sequencelist)
                        sequencelist = []


        sequencenum = len(self.imgnamelist)
        print 'Read', sequencenum, 'sequecnes...'

        total_seq_num = 0
        for sequ in self.imgnamelist:
            total_seq_num += len(sequ) - batch + 1
            self.episodeNum.append(total_seq_num)
        self.N = total_seq_num
        # print total_seq_num
        # print self.episodeNum

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        epiInd=0 # calculate the epiInd
        while idx>=self.episodeNum[epiInd]:
            # print self.episodeNum[epiInd],
            epiInd += 1 
        if epiInd>0:
            idx -= self.episodeNum[epiInd-1]

        # print epiInd, idx
        imgseq = []
        for k in range(self.batch):
            img = cv2.imread(self.imgnamelist[epiInd][idx+k])
            # print self.imgnamelist[epiInd][idx+k]
            
            if self.aug:
                img = im_hsv_augmentation(img)
                img = im_crop(img,maxscale=0.1)

<<<<<<< HEAD
            outimg = im_scale_norm_pad(img, outsize=192, down_reso=True, down_len=10)
=======
            outimg = im_scale_norm_pad(img, outsize=self.imgsize, mean=self.mean, std=self.std, down_reso=True)
>>>>>>> ab6beac641a82228dcfa8448556aa37e87fb7898

            imgseq.append(outimg)

        return np.array(imgseq)


if __name__=='__main__':
    # test 
    np.set_printoptions(precision=4)
    trackingUnlabelDataset = TrackingUnlabelDataset(batch=10, data_aug=True)
    for k in range(1):
        imgseq = trackingUnlabelDataset[k*1000]
        print imgseq.dtype, imgseq.shape
        seq_show(imgseq, scale=0.8)
        # cv2.imshow('img',facingDroneUnlabelDataset.img_denormalize(imgseq[5,:,:,:]))
        # cv2.waitKey(0)

    dataloader = DataLoader(trackingUnlabelDataset, batch_size=1, shuffle=True, num_workers=1)

    dataiter = iter(dataloader)
   
    while True:
        try:
            sample = dataiter.next()
        except:
            dataiter = iter(dataloader)
            sample = dataiter.next()

        seq_show(sample.squeeze().numpy(), scale=0.8)