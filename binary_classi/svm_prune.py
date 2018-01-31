#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: feiyang
"""

from cvxopt.solvers import qp
from cvxopt.base import matrix

import numpy, pylab, random, math
import numpy as np
import loaddata as ldata
import matplotlib.pyplot as plt
import time
    
def kernel(xm,xn):
    #r is the 'width' parameter
    r = 0.5
    dist = (xm[0] - xn[0])**2 + (xm[1] - xn[1])**2
    k = np.exp(-r**(-2) * dist)
    return k
    
    
def MatrixP(x):
    n = len(x)
    P = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            P[i, j] = x[i][2]*x[j][2]*kernel(x[i][0:2], x[j][0:2])
    return P
    
def alphaf(x, P, slack, C):
  n = len(x)
  q = -1*np.ones((n,1))
  if (slack == 0):
    h = np.zeros((n,1))
    G = -np.eye(n)
  else:
    h = np.vstack((np.zeros((n,1)),C * np.ones((n,1))))
    G = np.vstack((-np.eye(n),np.eye(n)))

  r = qp(matrix(P),matrix(q),matrix(G),matrix(h))
  alpha = list(r['x'])

  #find non-zero
  m = len(alpha)
  vectors = list()

  for i in range(m):
      if (slack == 1):
          if (alpha[i] > 1e-5) and (alpha[i] < C) :
              vectors.append((x[i][0], x[i][1], x[i][2], alpha[i]))
      else:
          if (alpha[i] > 1e-5):
              vectors.append((x[i][0], x[i][1], x[i][2], alpha[i]))
  return vectors


def ind(xstar,vectors):
   ind = 0
   n = len(vectors)
   for i in range(n):
       #########
       ind += vectors[i][3] * vectors[i][2] * kernel(xstar,vectors[i][0:2])
   return(ind)
    
def drawfun(vectors):
    xrange = np.linspace(-1.3, 0.75, 100)
    yrange = np.linspace(-0.2, 1.1, 100)
    grid = matrix([[ind([x, y], vectors) for y in yrange] for x in xrange])
    #plt.contour(xrange, yrange, grid, (-1.0, 0.0, 1.0), colors = ('red', 'black', 'blue') , linewidths = (1, 3, 1))
    plt.contour(xrange, yrange, grid, 0.0, colors = 'black', linestyles = 'dashed', linewidths =  3)

def prune(vector):
    n = len(vector)
    threshold = 10**-10
    sv = []
    sv_x = []
    sv_y = []
    for i in range(n):
        if abs(vector[i][3]) > threshold:
            sv.append(vector[i])
            sv_x.append(vector[i][0])
            sv_y.append(vector[i][1])

    return sv,sv_x,sv_y

def cal_error(data,vector):
    n = len(data)
    output = []
    error = 0
    for i in range(n):
        out = ind([data[i][0],data[i][1]], vector)
        if (out * data[i][2] < 0):
            error += 1.0

    return error/n

start = time.clock()
#main

group1 = ldata.group1
group2 = ldata.group2
x = ldata.x
y = ldata.y
classA = [(x[group1[i]],y[group1[i]],-1) for i in range(len(group1))]
classB = [(x[group2[j]],y[group2[j]],1) for j in range(len(group2))]
data = classA + classB

P = MatrixP(data)
slack = 1
vec = alphaf(data, P, slack, C=1)
sv,sv_x,sv_y = prune(vec)
error_train = cal_error(data,sv)
print('error rate for train dataset in SVM is: ' + str(error_train))
print('the number of data is: '+str(len(data)))
print('the number of supporting vector is: '+str(len(sv)))

#plot
plt.figure()
plt.subplot(1,2,1)
dot1 = plt.scatter(x[group1],y[group1], marker='o', c='k')
dot2 = plt.scatter(x[group2],y[group2], marker='x', c='b')
boundary = drawfun(vec)
scatter1 = plt.scatter(sv_x,sv_y, marker='o',c='',edgecolors='r',linewidths = 2)
plt.title('train data with true label in SVM:')

#for test data
test_group1 = ldata.test_group1
test_group2 = ldata.test_group2
test_x = ldata.test_x
test_y = ldata.test_y
test_classA = [(test_x[test_group1[i]],test_y[test_group1[i]],-1) for i in range(len(test_group1))]
test_classB = [(test_x[test_group2[j]],test_y[test_group2[j]],1) for j in range(len(test_group2))]
test_data = test_classA + test_classB

error_test = cal_error(test_data,sv)
print('error rate for test dataset is: ' + str(error_test))

elapsed = (time.clock() - start)
print("Time used:",elapsed)

plt.subplot(1,2,2)
dot3 = plt.scatter(test_x[test_group1],test_y[test_group1], marker='o', c='k')
dot4 = plt.scatter(test_x[test_group2],test_y[test_group2], marker='x', c='b')
boundary = drawfun(vec)
scatter2 = plt.scatter(sv_x,sv_y, marker='o',c='',edgecolors='r',linewidths = 2)
plt.title('test data with true label in SVM')

plt.show()







