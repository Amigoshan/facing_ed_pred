import cv2
import numpy as np

from os.path import isfile, join, isdir
from os import listdir
import xml.etree.ElementTree
from torch.utils.data import Dataset, DataLoader
from utils import im_scale_norm_pad, img_denormalize, im_hsv_augmentation, im_crop

import matplotlib.pyplot as plt

class XmlReader:
    def __init__(self):
        # self.dir2ind = {'n': 0,'ne': 1,'e': 2, 'se': 3,'s': 4,'sw': 5,'w': 6,'nw': 7}
        pass

    def readxml(self, xmldir):
        tree = xml.etree.ElementTree.parse(xmldir)
        root = tree.getroot()
        objects = root.findall("object")
        objects_list = []
        for obj in objects:
            xmin = obj.findall('bndbox/xmin')[0].text
            xmax = obj.findall('bndbox/xmax')[0].text
            ymin = obj.findall('bndbox/ymin')[0].text
            ymax = obj.findall('bndbox/ymax')[0].text
            objects_list.append([float(xmin),float(ymin),float(xmax),float(ymax)])
        return objects_list


def fileInDir(dirname):

    if isdir(dirname):
        for filename in listdir(dirname):
            filepathname = join(dirname, filename)
            if isfile(filepathname):
                print 'find file: ', filename
                yield filename



class CocoDataset(Dataset):

    def __init__(self, annodir = '/datadrive/coco/car_heading_2018/annotations', 
                        imgdir='/datadrive/coco/car_heading_2018/train',
                        imgsize = 192, 
                        data_aug=False,
                        mean=[0,0,0],std=[1,1,1]):

        self.imgsize = imgsize
        self.imgnamelist = []
        self.bboxlist = []
        self.aug = data_aug
        self.mean = mean
        self.std = std

        xmlreader = XmlReader()

        imgind = 0
        for xmlfile in fileInDir(annodir):
            xmlfilepath = join(annodir, xmlfile)

            objlist = xmlreader.readxml(xmlfilepath)
            if len(objlist)<=0:
                print 'no valid bbox in:', xmlfile
                continue

            datayear = xmlfile.split('_')[1]
            if datayear=='train2014':
                imagefilepath = join(imgdir, xmlfile.split('.')[0]+'.jpg')
            elif datayear=='train2017':
                imagefilepath = join(imgdir, xmlfile.split('.')[0].split('_')[-1]+'.jpg')
            else:
                print 'wrong datayear:', datayear
                continue

            if isfile(imagefilepath): # the image also valid
                for obj in objlist:
                    self.bboxlist.append(np.array(obj).astype(np.int))

                    self.imgnamelist.append(imagefilepath)
            else:
                print 'missing file:', imagefilepath
        self.N = len(self.imgnamelist)
        print 'Read', self.N, 'valid bboxes...'

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        img = cv2.imread(self.imgnamelist[idx]) # in bgr
        bbox = self.bboxlist[idx]

        img = img[bbox[1]:bbox[3],bbox[0]:bbox[2],:] # crop the bbox

        if self.aug:
            img = im_hsv_augmentation(img)
            img = im_crop(img,maxscale=0.08)


        outimg = im_scale_norm_pad(img, outsize=self.imgsize, mean=self.mean, std=self.std, down_reso=True)

        return {'img':outimg}

if __name__=='__main__':
    # test 
    np.set_printoptions(precision=4)
    cocoDataset = CocoDataset(data_aug=True)
    for k in range(100):
        img = cocoDataset[k]['img']
        print img.dtype#, CocoDataset[k]['label']
        print np.max(img), np.min(img), np.mean(img)
        print img.shape
        img = img_denormalize(img)
        cv2.imshow('img',img)
        cv2.waitKey(0)

