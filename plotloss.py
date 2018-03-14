import matplotlib.pyplot as plt
import numpy as np
from utils import groupPlot
from os.path import join

logname = '8_1_loss'
imgoutdir = 'resimg_facing'
AvgNum = 100
LOGFILE = False

datadir = 'data_facing'
exp_pref = '10_11_'
plotnum = -1

if LOGFILE:

	with open(join('log',logname+'.log')) as f:
		lines=f.readlines()

	trainloss=[]
	valloss=[]
	unlabelloss=[]
	for line in lines:
		ind=line.find('label-loss:')
		# print ind
		if ind>0: 
			line = line[ind+11:].strip()
			loss = line.split(',')[0]
			trainloss.append(float(loss))
		ind=line.find('val-loss:')
		if ind>0:
			# print ind
			line = line[ind+9:].strip()
			loss = line.split(',')[0]
			# line = line[ind1+13,:]
			# print loss
			valloss.append(float(loss))
		ind=line.find('unlabel-loss:')
		if ind>0:
			# print ind
			line = line[ind+13:].strip()
			loss = line.split(',')[0]
			# line = line[ind1+13,:]
			# print loss
			unlabelloss.append(float(loss))

else: 
	trainloss = np.load(join(datadir,exp_pref+'lossplot.npy'))
	valloss = np.load(join(datadir,exp_pref+'vallossplot.npy'))
	unlabelloss = np.load(join(datadir,exp_pref+'unlabellossplot.npy'))

if np.sum(valloss[0:9])==0:
			valloss[0:9]=valloss[10:19]

print valloss[0:100]

trainloss = np.array(trainloss[0:plotnum])
valloss = np.array(valloss[0:plotnum])
unlabelloss = np.array(unlabelloss[0:plotnum])

# print 'train: %.5f, val: %.5f, unlabel: %.5f' % (np.mean(trainloss[-AvgNum:]), np.mean(valloss[-AvgNum:]), np.mean(unlabelloss[-AvgNum:]))
# print '%.2f, %.2f, %.2f' % (np.mean(trainloss[-AvgNum:]), np.mean(valloss[-AvgNum:]), np.mean(unlabelloss[-AvgNum:]))

ax1 = plt.subplot(121)
ax1.plot(trainloss,label='training loss')
ax1.plot(valloss,label='validation loss')
ax1.legend()
ax1.grid()
ax1.set_ylim(0,1)
gpunlabelx, gpunlabely = groupPlot(range(len(trainloss)),trainloss,group=1000)
ax1.plot(gpunlabelx, gpunlabely,'y')
gpunlabelx, gpunlabely = groupPlot(range(len(valloss)),valloss,group=1000)
ax1.plot(gpunlabelx, gpunlabely,'y')


ax2 = plt.subplot(122)
gpunlabelx, gpunlabely = groupPlot(range(len(unlabelloss)),unlabelloss,group=1000)
ax2.plot(unlabelloss,label='continuity loss')
ax2.plot(gpunlabelx, gpunlabely, color='y', label='average')
ax2.legend()
ax2.grid()
ax2.set_ylim(0,10)

if LOGFILE:
	sn = logname
else:
	sn = exp_pref
plt.savefig(join(imgoutdir, sn+'.png'))

plt.show()