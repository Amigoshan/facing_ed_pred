import matplotlib.pyplot as plt
import numpy as np
from utils import groupPlot
from os.path import join

logname = '10_2_loss'
imgoutdir = 'resimg_facing'
AvgNum = 100
LOGFILE = False

datadir = 'data_facing'
exp_pref = ['21_11_','21_8_','21_10_']
# plotnum = -1
legendlist = ['128 hidden units','64 hidden units','32 hidden units','supervised only']

ax1 = plt.subplot(121)
ax2 = plt.subplot(122)

for k,exp in enumerate(exp_pref):
	trainloss = np.load(join(datadir,exp+'labellossplot.npy'))
	valloss = np.load(join(datadir,exp+'vallossplot.npy'))
	if k<3:
		unlabelloss = np.load(join(datadir,exp+'unlabellossplot.npy'))

	if np.sum(valloss[0:49])==0:
				valloss[0:49]=valloss[50:99]


	trainloss = np.array(trainloss)
	valloss = np.array(valloss)
	if k<3:
		unlabelloss = np.array(unlabelloss)

	gpunlabelx, gpunlabely = groupPlot(range(len(valloss)),valloss,group=1000)
	print len(gpunlabely)
	ax1.plot(gpunlabelx, gpunlabely,label=legendlist[k])
	ax1.legend()
	ax1.grid()

	if k<3:
		gpunlabelx, gpunlabely = groupPlot(range(len(unlabelloss)),unlabelloss,group=1000)
		ax2.plot(gpunlabelx, gpunlabely,label=legendlist[k])
		ax2.legend()
		ax2.grid()

logname = '8_1_loss'
imgoutdir = 'resimg_facing'
AvgNum = 100

datadir = 'data_facing'
exp_pref = '21_4_'
plotnum = -1


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
print len(valloss)
gpunlabelxx, gpunlabely = groupPlot(range(len(valloss)),valloss,group=50)
ax1.plot(gpunlabelx, gpunlabely,label=legendlist[3])
ax1.legend()
ax1.grid()
ax1.set_ylim(0.375,0.5)

# print 'train: %.5f, val: %.5f, unlabel: %.5f' % (np.mean(trainloss[-AvgNum:]), np.mean(valloss[-AvgNum:]), np.mean(unlabelloss[-AvgNum:]))
# print '%.2f, %.2f, %.2f' % (np.mean(trainloss[-AvgNum:]), np.mean(valloss[-AvgNum:]), np.mean(unlabelloss[-AvgNum:]))


# ax1.plot(trainloss,label='training loss')
# ax1.plot(valloss,label='validation loss')
# ax1.legend()
# ax1.grid()
# ax1.set_ylim(0,1)
# gpunlabelx, gpunlabely = groupPlot(range(len(trainloss)),trainloss,group=1000)
# ax1.plot(gpunlabelx, gpunlabely,'y')
# gpunlabelx, gpunlabely = groupPlot(range(len(valloss)),valloss,group=1000)
# ax1.plot(gpunlabelx, gpunlabely,'y')



# gpunlabelx, gpunlabely = groupPlot(range(len(unlabelloss)),unlabelloss,group=1000)
# ax2.plot(unlabelloss,label='continuity loss')
# ax2.plot(gpunlabelx, gpunlabely, color='y', label='average')
# ax2.legend()
# ax2.grid()
# ax2.set_ylim(0,1)

# if LOGFILE:
# 	sn = logname
# else:
# 	sn = exp_pref
# plt.savefig(join(imgoutdir, sn+'.png'))
ax1.grid()
plt.show()