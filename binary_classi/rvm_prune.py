"""
basic rvm classify function
classify points from two datasets, and draw the boundary

@author: Feiyang Liu
"""

import numpy as np
import matplotlib.pyplot as plt
import loaddata as ldata
import pylab
from numpy import random
import random as rand
import time

##RVM
def kernel(xm,xn):
#r is the 'width' parameter
	r = 0.5
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
	for i in range(n):
		A[i,i] = (1 - A[i,i]*Sigma[i,i]) / (w[i]**2+NOT_ZERO)

	return A

def cal_error(label,output):
	error = 0.0
	total = output.size
	for i in range(total):
		a = label[i] * output[i]
		if a < 0:
		#if label[i] == 1 != output[i] > np.exp(-1):
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

def draw_boundary(w,train):
	x = np.array(np.linspace(-1.3,0.75,100))
	y = np.array(np.linspace(-0.2,1.1,100))
	grid = np.zeros((100,100))
	n = len(w)
	for i in range(100):
		for j in range(100):
			data = np.array([x[i],y[j]])
			phi = np.zeros(n)
			phi[0] = 1
			for z in range(n - 1):
				phi[z + 1] = kernel(data,train[z])
			grid[i,j] = np.dot(phi,w)
	#plt.contour(x, y, grid, (-1, 0.0, 1), colors = ('red', 'black', 'blue') , linewidths = (1, 3, 1))
	plt.contour(x, y, grid, 0.0, colors = 'black', linestyles = 'dashed', linewidths =  3)

def prune(w,Rec,phi,alpha):
	new_w = w[Rec]
	new_phi = phi[:,Rec]
	new_alpha = alpha[Rec]
	new_A = np.diag(new_alpha)

	return new_w, new_phi, new_A

#main function
start = time.clock()

#load training data first
x = ldata.x
y = ldata.y
t = ldata.t
group1 = ldata.group1
group2 = ldata.group2

NUMPOINTS, = x.shape
NOT_ZERO = 10**-50

train_data = (np.array([x,y])).T
phi = cal_Phi(train_data)

#initialize
A = np.diag(random.rand(NUMPOINTS+1))
B = np.diag(random.rand(NUMPOINTS))
w = np.array(random.rand(NUMPOINTS+1))
Sigma = np.zeros((NUMPOINTS+1,NUMPOINTS+1))
Y = np.dot(phi,w)

error_train = 1
iter = 0

remaining_inx = np.indices(w.shape).reshape(-1,)

#begin training 
while (error_train > 10**-1 and iter < 2000):
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

#error_train = cal_error(t, Y)
print('error rate for train dataset in RVM is: ' + str(error_train))
print('the number of data is: '+str(NUMPOINTS))
print('the number of supporting vector is: '+str(len(R_vec)-1))
print('iteration times: '+str(iter))

plt.figure()

plt.subplot(1,2,1)
dot1 = plt.scatter(x[group1],y[group1], marker='o', c='k')
dot2 = plt.scatter(x[group2],y[group2], marker='x', c='b')
scatter1 = plt.scatter(x[remaining_inx[1:]-1],y[remaining_inx[1:]-1], marker='o',c='',edgecolors='r',linewidths = 2)
draw_boundary(w,train_data[remaining_inx[1:]-1])
plt.title('train data with true label in RVM')

# test
test_t = ldata.test_t
test_group1 = ldata.test_group1
test_group2 = ldata.test_group2
test_x = ldata.test_x
test_y = ldata.test_y

test_data = (np.array([test_x,test_y])).T
test_Y = cal_out(test_data, train_data[remaining_inx[1:]-1], w)
error_test = cal_error(test_t, test_Y)
print('error rate for test dataset is: ' + str(error_test))

elapsed = (time.clock() - start)
print("Time used:",elapsed)

plt.subplot(1,2,2)
dot3 = plt.scatter(test_x[test_group1],test_y[test_group1], marker='o', c='k')
dot4 = plt.scatter(test_x[test_group2],test_y[test_group2], marker='x', c='b')
scatter2 = plt.scatter(x[remaining_inx[1:]-1],y[remaining_inx[1:]-1], marker='o',c='',edgecolors='r',linewidths = 2)
draw_boundary(w,train_data[remaining_inx[1:]-1])
plt.title('test data with true label in RVM')

plt.show()


