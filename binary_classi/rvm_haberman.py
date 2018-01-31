'''
general rvm classify function
using for multi class classification
train n classifier if there are n classes

main function: rvm

@author: Feiyang Liu

'''
import numpy as np
import matplotlib.pyplot as plt
import pylab, math
from numpy import random
import random as rand

def kernel(xm,xn):
#r is the 'width' parameter
	r = -0.5
	dist = np.linalg.norm(xm - xn)
	k = np.exp(-r**(-2) * dist**2)
	return k


def sigma(y):
	return(1 / (1+np.exp(-y)))

def Cap_sigma(A,B,Phi):
	return np.linalg.inv(Phi.T @ B @ Phi + A)

def w_MP(Sigma,Phi,B,t):
	return Sigma @ Phi.T @ B @ t 

def beta(y):
	return sigma(y)*(1-sigma(y))

def cal_B(Y):
	B = []
	for i in range(NUMPOINTS):
		beta_i = beta(Y[i])
		B.append(beta_i)

	return np.diag(B)

def cal_Phi(data):
	rep = np.zeros((NUMPOINTS, NUMPOINTS + 1))
	for i in range(NUMPOINTS):
		rep[i, 0] = 1
		for j in range(NUMPOINTS):
			rep[i, j + 1] = kernel(data[i], data[j])
	return rep

def updata_A(Sigma,w,A):
	n = A.shape[0]
	NOT_ZERO = 10**-50
	for i in range(n):
		A[i,i] = (1 - A[i,i]*Sigma[i,i]) / (w[i]**2+NOT_ZERO)

	return A

def cal_correct(label,output):
	right = 0.0
	total = output.size
	for i in range(total):
		if (label[i] > 0 and output[i] > 0):
			right = right + 1.0

	return right

def cal_error(label,output):
	error = 0.0
	total = output.size
	for i in range(total):
		a = label[i] * output[i]
		if a < 0:
			error = error + 1.0

	return error/float(total)

def cal_out(test, train, w):
	test_num = test.shape[0]
	Y_out = np.zeros(test_num)
	n = len(w)
	for i in range(test_num):
		phi = np.zeros(n)
		phi[0] = 1
		for j in range(n-1):
			phi[j + 1] = kernel(test[i],train[j])
		Y_out[i] = np.dot(phi,w)

	return Y_out

def dis_kernel(xm,xn):
#r is the 'width' parameter
	r = 0.5
	dist = np.linalg.norm(xm - xn)
	k = np.exp(-r**(-2) * dist**2)
	return k


def prune(w,Rec,phi,alpha):
	new_w = w[Rec]
	new_phi = phi[:,Rec]
	new_alpha = alpha[Rec]
	new_A = np.diag(new_alpha)

	return new_w, new_phi, new_A

def train_rvm(train, t, class_num):

	phi = cal_Phi(train)

	A = np.diag(random.rand(NUMPOINTS+1))
	B = np.diag(random.rand(NUMPOINTS))
	w = np.array(random.rand(NUMPOINTS+1))
	Sigma = np.zeros((NUMPOINTS+1,NUMPOINTS+1))
	Y = np.dot(phi,w)

	right_train = 0
	iter = 0

	remaining_inx = np.indices(w.shape).reshape(-1,)
	threshold = 1.0 / float(class_num) * 0.9

#begin training 
	while (right_train < threshold and iter < 2000):
		B = cal_B(Y)
		Sigma = Cap_sigma(A,B,phi)
		w = w_MP(Sigma,phi,B,t)
		A = updata_A(Sigma,w,A)
		alpha = np.diag(A)
		R_vec = [i for i,v in enumerate(alpha) if abs(v) < 10**12]
		if R_vec[0] != 0:
			R_vec = [0] + R_vec
		remaining_inx = remaining_inx[R_vec]
		w, phi, A =  prune(w,R_vec,phi,alpha)
		Y = np.dot(phi,w)
		right_classify = cal_correct(t, Y)
		right_train = right_classify / float(NUMPOINTS)
		iter += 1


	return right_classify,w

def train_rvm_2(train, t, class_num=2):

	phi = cal_Phi(train)

	A = np.diag(random.rand(NUMPOINTS+1))
	B = np.diag(random.rand(NUMPOINTS))
	w = np.array(random.rand(NUMPOINTS+1))
	Sigma = np.zeros((NUMPOINTS+1,NUMPOINTS+1))
	Y = np.dot(phi,w)

	iter = 0
	error_train = 1

	remaining_inx = np.indices(w.shape).reshape(-1,)

#begin training 
	while (error_train > 10**-1 and iter < 100):
		B = cal_B(Y)
		Sigma = Cap_sigma(A,B,phi)
		w = w_MP(Sigma,phi,B,t)
		A = updata_A(Sigma,w,A)
		alpha = np.diag(A)
		R_vec = [i for i,v in enumerate(alpha) if abs(v) < 10**12]
		if R_vec[0] != 0:
			R_vec = [0] + R_vec
		remaining_inx = remaining_inx[R_vec]
		w, phi, A =  prune(w,R_vec,phi,alpha)
		Y = np.dot(phi,w)
		error_train = cal_error(t, Y)
		iter += 1

	return error_train, w, remaining_inx

def cal_out(test, train, w):
	test_num = test.shape[0]
	Y_out = np.zeros(test_num)
	n = len(w)
	for i in range(test_num):
		phi = np.zeros(n)
		phi[0] = 1
		for j in range(n-1):
			phi[j + 1] = kernel(test[i],train[j])
		Y_out[i] = np.dot(phi,w)

	return Y_out


def rvm(train, test, train_label, test_label, class_num = None):
	if class_num == None:
		label_order = list(set(train_label))  
		class_num = len(label_order) 

	right_class = []
	weight = []
	if class_num > 2:
		for i in range(class_num):
			new_label = []
			for j in range(NUMPOINTS):
				index = (train_label[j] == label_order[i]) * 2 - 1
				new_label.append(index)

			right_classify,w = train_rvm(train, new_label, class_num)
			print(right_classify)
			right_class.append(right_classify)
			weight.append(w)

		train_error = 1.0 - sum(right_class) / float(len(train_label))
		test_error = None
	else:
		train_error, w, remaining_inx = train_rvm_2(train, train_label, class_num=2)
		test_Y = cal_out(test_data, train_data[remaining_inx[1:]-1], w)
		test_error = cal_error(test_label, test_Y)

	return train_error,test_error,len(w) - 1

#main
'''
import loadimage as loadi

train_image = loadi.output_ims[0:5000,:]
train_label = loadi.output_labels[0:5000]

NUMPOINTS = len(train_label)
NOT_ZERO = 10**-50

train_error = rvm(train_image, train_label, class_num = None)
print('training error rate for nmist dataset is:' + str(train_error))
'''
import loaddata_har as ldata

train_label = ldata.t[0:200]
test_label = ldata.t[200:300]
train_data = ldata.data[0:200,:]
test_data = ldata.data[200:300,:]

NUMPOINTS = len(train_label)
NOT_ZERO = 10**-50

train_error, test_error, RV = rvm(train_data, test_data, train_label, test_label, class_num = None)
print('training error rate for haberman dataset is:' + str(train_error))
print('training error rate for haberman dataset is:' + str(train_error))
print('the number of supporting vector is:' + str(RV))




	