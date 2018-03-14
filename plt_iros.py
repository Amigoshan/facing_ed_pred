import matplotlib.pyplot as plt
import numpy as np
# from utils import groupPlot
from os.path import join

def groupPlot(datax, datay, group=10):
    datax, datay = np.array(datax), np.array(datay)
    if len(datax)%group>0:
        datax = datax[0:len(datax)/group*group]
        datay = datay[0:len(datay)/group*group]
    datax, datay = datax.reshape((-1,group)), datay.reshape((-1,group))
    datax, datay = datax.mean(axis=1), datay.mean(axis=1)
    return (datax, datay)


datadir = '/datadrive/aayush/data_facing'
filelist = [#['track_sup_run2_wo_norm_lossplot.npy','track_sup_run2_wo_norm_vallossplot.npy'],
			['track_sup_seed_drone_label_run3_lossplot.npy','track_sup_seed_drone_label_run3_vallossplot.npy'],
			['track_sup_seed_drone_unlabel_run3_lossplot.npy','track_sup_seed_drone_unlabel_run3_vallossplot.npy'],
			['track_sup_seed_drone_unlabel_coco_label_run3_lossplot.npy','track_sup_seed_drone_unlabel_coco_label_run3_vallossplot.npy'],
			['track_sup_seed_drone_both_run3_lossplot.npy','track_sup_seed_drone_both_run3_vallossplot.npy'],
			# ['track_sup_seed_drone_label_run2_lossplot.npy','track_sup_seed_drone_label_run2_vallossplot.npy'],
			# ['track_sup_seed_drone_unlabel_run2_lossplot.npy','track_sup_seed_drone_unlabel_run2_vallossplot.npy'],
			# ['track_sup_seed_drone_both_run2_lossplot.npy','track_sup_seed_drone_both_run2_vallossplot.npy']
			]
labellist = [['training loss','validation loss'],
			 ['training loss','validation loss'],
			 ['training loss','validation loss'],
			 ['training loss','validation loss'],			 
			 ]
titlelist = [#'Training on VIRAT dataset',
			 'Labeled only',
			 'Unlabeled only',
			 'Combined (different distribution)',
			 'Combined (same distribution)']
imgoutdir = 'resimg_facing'
AvgNum = 100

# for ind,files in enumerate(filelist):
# 	print ind, files
# 	ax=plt.subplot(int('22'+str(ind+1)))
# 	# lines = []

# 	for k,filename in enumerate(files):


# 		loss = np.load(join(datadir,filename))
# 		print loss.shape, np.sum(loss[0:49])
# 		if np.sum(loss[0:49])==0:
# 			loss[0:49]=loss[50:99]
# 		ax.plot(loss, label=labellist[ind][k])
# 		datax, datay = groupPlot(range(1,loss.shape[0]+1),loss, group=5000)
# 		ax.plot(datax, datay, 'y')


# 	ax.grid()
# 	ax.legend()
# 	ax.set_ylim(0,0.8)
# 	ax.set_xlabel('number of iterations')
# 	ax.set_ylabel('loss')
# 	ax.set_title(titlelist[ind])

# plt.show()

for ind,files in enumerate(filelist):
	print ind, files
	# ax=plt.subplot(int('22'+str(ind+1)))
	# lines = []

	# for k,filename in enumerate(files):


	loss = np.load(join(datadir,files[1]))
	print loss.shape, np.sum(loss[0:49])
	if np.sum(loss[0:49])==0:
		loss[0:49]=loss[50:99]
	# loss = loss[0:200000]
	plt.plot(loss, label=titlelist[ind])
	datax, datay = groupPlot(range(1,loss.shape[0]+1),loss, group=5000)
	plt.plot(datax, datay, 'y')


plt.grid()
plt.legend()
plt.ylim(0.2,0.8)
plt.xlabel('number of iterations')
plt.ylabel('loss')
	# plt.title(titlelist[ind])

plt.show()

