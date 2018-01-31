"""

@author: feiyang
"""

import numpy as np
from numpy import random


def readdata(filename):
	x = []
	y = []
	label = []
	with open(filename, 'r') as file_to_read:
		while True:
			lines = file_to_read.readline()
			if not lines:
				break
			pass
			x_tmp, y_tmp, label_tmp = [float(i) for i in lines.split()]
			x.append(x_tmp)
			y.append(y_tmp)
			label.append(label_tmp)
			pass
		x = np.array(x)
		y = np.array(y)
		#label = np.array(label)
		pass

	group1 = [i for i,v in enumerate(label) if v==0]
	group2 = [i for i,v in enumerate(label) if v==1]
	#t = label
	t = [num*2 - 1 for num in label]
	t = np.array(t).T

	return x,y,label,t,group1,group2

#training data
filename = './dataset/synth_tr.txt'
x,y,label,t,group1,group2 = readdata(filename)

# test
filename1 = './dataset/synth_te.txt'
test_x,test_y,test_label,test_t,test_group1,test_group2 = readdata(filename1)
