import cv2
import numpy as np

from os.path import isfile, join, isdir
from os import listdir
import xml.etree.ElementTree
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt


class FacingDroneUnlabelDataset(Dataset):

    def __init__(self, imgdir='/home/wenshan/datasets/droneData/',imgsize = 192, batch = 32):

        self.imgsize = imgsize
        self.imgnamelist = []
        self.fileprefix = 'drone_'
        self.folderlist = ['4','7','11','17','23','30','32','33','37','38','49','50']
        self.maxind = [4918,4100, 5265, 7550, 4157, 6350, 6492, 8402, 5131, 4907,2574, 3140]
        self.batch = batch
        self.episodeNum = []

        for f_ind, foldername in enumerate(self.folderlist):

            folderpath = join(imgdir, foldername)
            # filenum = len(listdir(folderpath))
            # print 'find %d entities in %s ' % (filenum, folderpath) 

            sequencelist = []
            missimg = 0
            for imgind in range(0, self.maxind[f_ind]):
                filename = self.fileprefix+foldername+'_'+str(imgind)+'.jpg'
                filepathname = join(folderpath, filename)
                if isfile(filepathname):
                    sequencelist.append(filepathname)
                    if missimg>0:
                        # print '  -- last missimg', missimg
                        missimg = 0
                else:
                    if len(sequencelist)>0:
                        missimg = 1
                        self.imgnamelist.append(sequencelist)
                        # print 'image lost:', filename
                        # print '** sequence: ', len(sequencelist)
                        sequencelist = []
                    else:
                        missimg += 1
            if len(sequencelist)>0:          
                self.imgnamelist.append(sequencelist)
                # print '** sequence: ', len(sequencelist)
                sequencelist = []


        sequencenum = len(self.imgnamelist)
        print 'Read', sequencenum, 'sequecnes...'

        total_seq_num = 0
        for sequ in self.imgnamelist:
            total_seq_num += len(sequ) - batch + 1
            self.episodeNum.append(total_seq_num)
        self.N = total_seq_num
        print total_seq_num
        print self.episodeNum

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
            imgseq.append(outimg)

        return np.array(imgseq)

if __name__=='__main__':
    # test 
    np.set_printoptions(precision=4)
    facingDroneUnlabelDataset = FacingDroneUnlabelDataset()
    for k in range(1):
        imgseq = facingDroneUnlabelDataset[k*1000]
        print imgseq.dtype, imgseq.shape
        cv2.imshow('img',facingDroneUnlabelDataset.img_denormalize(imgseq[5,:,:,:]))
        cv2.waitKey(0)

    dataloader = DataLoader(facingDroneUnlabelDataset, batch_size=1, shuffle=True, num_workers=1)

    for sample in dataloader:
      print sample.size()
      
