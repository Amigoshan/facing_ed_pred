import torch
import math
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
	r = np.clip(1-math.sqrt(float(x1*x1+y1*y1))/t,0,1)
	x1, y1 = maxx-x, y
	g = np.clip(1-math.sqrt(float(x1*x1+y1*y1))/t,0,1)
	x1, y1 = x, maxy-y
	b = np.clip(1-math.sqrt(float(x1*x1+y1*y1))/t,0,1)
	# x1, y1 = maxx-x, maxy-y
	# a = math.sqrt(float(x1*x1+y1*y1))/t
	a = 1
	return (r,g,b,a)
