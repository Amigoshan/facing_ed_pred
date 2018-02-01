import cv2
from os.path import join, isfile, isdir
from os import listdir
import numpy as np
from torch.utils.data import Dataset, DataLoader

def imgInDir(dirname):
    if isdir(dirname):
        for filename in listdir(dirname):
            filepathname = join(dirname, filename)
            if isfile(filepathname):
                # print 'find image: ', filename
                # image = cv2.imread(filename)
                if filename[-3:] == 'jpg':
                    yield filepathname

class FacingDataset(Dataset):

    def __init__(self, datadir = '/home/wenshan/datasets/facing',imgsize = 192):

        dirlist = ['Session1','Session2','Session3']
        self.imgsize = imgsize
        self.imgnamelist = []

        imgind = 0
        for predir in dirlist:
            prepath = join(datadir, predir)
            for subdir in listdir(prepath):
                subpath = join(prepath, subdir)
                if isdir(subpath):
                    for img in imgInDir(subpath):
                        self.imgnamelist.append(img)

                        imgind += 1 

        self.N = imgind

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
        resize_scale = float(self.imgsize)/np.max(img.shape)
        img = cv2.resize(img, (0,0), fx = resize_scale, fy = resize_scale)
        img = self.img_normalize(img)
        # print img.shape
        startx = (self.imgsize-img.shape[2])/2
        starty = (self.imgsize-img.shape[1])/2
        # print startx, starty
        outimg = np.zeros((3,self.imgsize,self.imgsize), dtype=np.float32)
        outimg[:,starty:starty+img.shape[1], startx:startx+img.shape[2]] = img

        return outimg

if __name__=='__main__':
    # test 
    np.set_printoptions(precision=4)
    facingDataset = FacingData()
    for k in range(100):
        img = facingDataset[k*10]
        print img.dtype
        print np.max(img), np.min(img), np.mean(img)
        print img.shape
        # cv2.imshow('img',img)
        # cv2.waitKey(0)
