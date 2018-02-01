import cv2
import torch
from os.path import isfile, join, split, isdir
import numpy as np
from torch.utils.data import Dataset, DataLoader
import codecs

class MnistManiDataset(Dataset):
    """2D map dataset. """


    def __init__(self, cat = 'train', warping=False, warp_scale = 0.2, fix_warping=False):
        datadir = 'data'
        train_data_file = 'train-images-idx3-ubyte'
        train_label_file = 'train-labels-idx1-ubyte'
        test_data_file = 't10k-images-idx3-ubyte'
        test_label_file = 't10k-labels-idx1-ubyte'

        self.cat = cat
        self.warping = warping # warping randomly for all 8 numbers from [-scale/2, scale/2]
        self.warp_scale = warp_scale 
        self.fix_warping = fix_warping # only warp one number to 'scale'
        if cat=='train':
            self.data = self.read_image_file(join(datadir, train_data_file))
            self.label = self.read_label_file(join(datadir, train_label_file))
        else:
            self.data = self.read_image_file(join(datadir, test_data_file))
            self.label = self.read_label_file(join(datadir, test_label_file))

        self.data = np.array(self.data, dtype=np.float32)/255.0
        self.label = np.array(self.label, dtype=np.int)

        self.N = len(self.data)

    def get_int(self, b):
        return int(codecs.encode(b, 'hex'), 16)


    def parse_byte(self, b):
        if isinstance(b, str):
            return ord(b)
        return b

    def read_label_file(self, path):
        with open(path, 'rb') as f:
            data = f.read()
            assert self.get_int(data[:4]) == 2049
            length = self.get_int(data[4:8])
            labels = [self.parse_byte(b) for b in data[8:]]
            assert len(labels) == length
            return labels


    def read_image_file(self, path):
        with open(path, 'rb') as f:
            data = f.read()
            assert self.get_int(data[:4]) == 2051
            length = self.get_int(data[4:8])
            num_rows = self.get_int(data[8:12])
            num_cols = self.get_int(data[12:16])
            images = []
            idx = 16
            for l in range(length):
                img = []
                images.append(img)
                for r in range(num_rows):
                    row = []
                    img.append(row)
                    for c in range(num_cols):
                        row.append(self.parse_byte(data[idx]))
                        idx += 1
            assert len(images) == length
            return images    

    def warpimg(self, img, warpvec):
        # warpvec: [a, b, c, d, e, f, c1, c2]
        # [ 1+a,    b,      e*10  
        #   c,      1+d,    f*10
        #   c1/100, c2/100, 1]
        M = np.array([[1+warpvec[0], warpvec[1], warpvec[4]*10.0],
                      [warpvec[2], 1+warpvec[3], warpvec[5]*10.0],
                      [warpvec[6]/100.0, warpvec[7]/100.0,    1]])
        dst = cv2.warpPerspective(img,M,(28,28))
        return dst

    def showdigit(self, img, time=0):
        if isinstance(img, torch.Tensor): # convert tensor to numpy
            img_show = img.numpy()
        else:
            img_show = img
        img_show = img_show.squeeze()
        if len(img_show.shape)==3: # a batch of imgs (in n x h x w), show them together
            img_show = img_show.transpose(1,0,2).reshape(img_show.shape[1], -1)
        cv2.imshow('digit', img_show)
        cv2.waitKey(time)
        cv2.destroyWindow('digit')  

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        warpimg = None
        warpvec = None
        if self.warpimg:
            warpvec = (np.random.rand(8) - 0.5)* self.warp_scale
            warpimg = self.warpimg(self.data[idx,:,:], warpvec)
        if self.fix_warping: 
            warpvec = np.zeros(8,dtype=np.float32)
            warpvec[np.random.randint(0,8)] = self.warp_scale * (np.random.randint(0,2)*2-1)
            warpimg = self.warpimg(self.data[idx,:,:], warpvec)

        return {'data': self.data[idx,:,:], 'label': self.label[idx], 'wdata': warpimg, 'warpvec': warpvec}


if __name__=='__main__':
    # test 
    np.set_printoptions(precision=4)
    mnistManiDataset = MnistManiDataset(cat='test', fix_warping=True, warp_scale=0.3)
    for k in range(100):
        sample = mnistManiDataset[k*101]
        img = sample['data']
        imgwarp = sample['wdata']
        print sample['label'], sample['warpvec']
        mnistManiDataset.showdigit(np.array([img,imgwarp]))
        # warpvec = (np.random.rand(8) - 0.5)/2
        # print warpvec
        # warpimg = mnistManiDataset.warpimg(img, warpvec)
        # mnistManiDataset.showdigit(np.array([img,warpimg]))



    dataloader = DataLoader(mnistManiDataset, batch_size=4, shuffle=True, num_workers=4)

    for sample in dataloader:
      print sample['data'].size(),sample['label'].size(), sample['label']
      mnistManiDataset.showdigit(sample['data'])

