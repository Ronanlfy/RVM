"""

@author: feiyang
"""

import numpy as np
from numpy import random



def readdata(filename):
	age = []
	year = []
	node = []
	label = []
	with open(filename, 'r') as file_to_read:
		while True:
			lines = file_to_read.readline()
			if not lines:
				break
			pass
			age_tmp, year_tmp, node_tmp, label_tmp = [float(i) for i in lines.split(',')]
			age.append(age_tmp)
			year.append(year_tmp)
			node.append(node_tmp)
			label.append(label_tmp)
			pass
		age = np.array(age)
		year = np.array(year)
		node = np.array(node)
		#label = np.array(label)
		pass

	t = [-2 * num + 3 for num in label]
	t = np.array(t).T
	data = (np.array([age,year,node])).T

	return data,label,t


#training data
filename = './dataset/haberman_data.txt'
data,label,t = readdata(filename)


