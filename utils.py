import cv2
import torch
from math import sqrt, sin, cos
import numpy as np
# import torch.nn as nn

def loadPretrain(model, preTrainModel):
	preTrainDict = torch.load(preTrainModel)
	model_dict = model.state_dict()
	print 'preTrainDict:',preTrainDict.keys()
	print 'modelDict:',model_dict.keys()
	preTrainDict = {k:v for k,v in preTrainDict.items() if k in model_dict}
	for item in preTrainDict:
		print '  Load pretrained layer: ',item
	model_dict.update(preTrainDict)
	# for item in model_dict:
	# 	print '  Model layer: ',item
	model.load_state_dict(model_dict)
	return model

def loadPretrain2(model, preTrainModel):
	preTrainDict = torch.load(preTrainModel)
	model_dict = model.state_dict()
	# print 'preTrainDict:',preTrainDict.keys()
	# print 'modelDict:',model_dict.keys()
	# update the keyname according to the last two words
	loadDict = {}
	for k,v in preTrainDict.items():
		keys = k.split('.')
		for k2,v2 in model_dict.items():
			keys2 = k2.split('.')
			if keys[-1]==keys2[-1] and (keys[-2]==keys2[-2] or 
				(keys[-2][1:]==keys2[-2][2:] and keys[-2][0]=='d' and keys2[-2][0:2]=='de')): # compansate for naming bug
				loadDict[k2]=v
				print '  Load pretrained layer: ',k2
				break

	model_dict.update(loadDict)
	# for item in model_dict:
	# 	print '  Model layer: ',item
	model.load_state_dict(model_dict)
	return model


def getColor(x,y,maxx,maxy):
	y = y*maxx/maxy
	maxy = maxx # normalize two axis
	x1, y1, t = x, y, maxx
	r = np.clip(1-sqrt(float(x1*x1+y1*y1))/t,0,1)
	x1, y1 = maxx-x, y
	g = np.clip(1-sqrt(float(x1*x1+y1*y1))/t,0,1)
	x1, y1 = x, maxy-y
	b = np.clip(1-sqrt(float(x1*x1+y1*y1))/t,0,1)
	# x1, y1 = maxx-x, maxy-y
	# a = sqrt(float(x1*x1+y1*y1))/t
	a = 1
	return (r,g,b,a)

def img_normalize(img):
    img = img[:,:,[2,1,0]] # bgr to rgb
    img = img.astype(np.float32)/255.0
    img = img.transpose(2,0,1)
    return img

def img_denormalize(img):
    # print img.shape
    img = img.transpose(1,2,0)
    img = img.clip(0,1) # network can output values out of range
    img = (img*255).astype(np.uint8)
    img = img[:,:,[2,1,0]]
    return img

def seq_show(imgseq, scale = 0.3):
    # input a numpy array: n x 3 x h x w
    imgnum = imgseq.shape[0]
    imgshow = []
    for k in range(imgnum):
        imgshow.append(img_denormalize(imgseq[k,:,:,:])) # n x h x w x 3
    imgshow = np.array(imgshow)
    imgshow = imgshow.transpose(1,0,2,3).reshape(imgseq.shape[2],-1,3) # h x (n x w) x 3
    imgshow = cv2.resize(imgshow,(0,0),fx=scale,fy=scale)
    cv2.imshow('img',imgshow)
    cv2.waitKey(0)

def put_arrow(img, dir):
	# print type(img), img.dtype, img.shape
	img = img.copy()
	cv2.line(img, (66,150), (126,150), (0, 255, 0), 2)
	cv2.line(img, (96,120), (96,180), (0, 255, 0), 2)

	cv2.arrowedLine(img, (96,150), (int(96+40*dir[1]),int(150-40*dir[0])), (0, 0, 255), 4)

	return img

def seq_show_with_arrow(imgseq, dirseq, scale = 0.8):
    # imgseq: a numpy array: n x 3 x h x w
    # dirseq: a numpy array: n x 2
    imgnum = imgseq.shape[0]
    imgshow = []
    for k in range(imgnum):
    	img = img_denormalize(imgseq[k,:,:,:])
    	img = put_arrow(img, dirseq[k,:])
        imgshow.append(img) # n x h x w x 3
    imgshow = np.array(imgshow)
    imgshow = imgshow.transpose(1,0,2,3).reshape(imgseq.shape[2],-1,3) # h x (n x w) x 3
    imgshow = cv2.resize(imgshow,(0,0),fx=scale,fy=scale)
    cv2.imshow('img',imgshow)
    cv2.waitKey(0)

