import matplotlib.pyplot as plt
import numpy as np
with open('loss.log') as f:
	lines=f.readlines()

trainloss=[]
valloss=[]
for line in lines:
	ind1=line.find('loss:')
	# print ind1
	loss1 = line[ind1+6:ind1+13]
	line = line[ind1+13:]
	trainloss.append(float(loss1))
	ind1=line.find('val-loss:')
	# print ind1
	loss1 = line[ind1+10:ind1+17]
	# line = line[ind1+13,:]
	# print loss1
	valloss.append(float(loss1))
plt.plot(np.array(trainloss))
plt.plot(np.array(valloss))
plt.ylim(0,2)
plt.grid()
plt.show()