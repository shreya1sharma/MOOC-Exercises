# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 14:42:07 2018

@author: 0000016446351
"""
'''
Implement regularized logistic regression to predict whether microchips from a fabrication plant passes quality assurance (QA). During QA, each microchip goes through various tests to ensure it is functioning correctly.
Suppose you are the product manager of the factory and you have the test results for some microchips on two different tests. From these two tests, you would like to determine whether the microchips should be accepted or rejected. To help you make the decision, you have a dataset of test results on past microchips, from which you can build a logistic regression model.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

data  = pd.read_csv('C:/Users/0000016446351/Desktop/ML_assi/data/ex2data2.txt', header = None)
X = data.iloc[:, 0:2]
y = data.iloc[:,2]

# visualizing the data
mask = y == 1
passed = plt.scatter(X[mask][0].values, X[mask][1].values, c = 'R')
failed = plt.scatter(X[~mask][0].values, X[~mask][1].values)
plt.xlabel('Microchip Test1')
plt.ylabel('Microchip Test2')
plt.legend((passed, failed), ('Passed', 'Failed'))
plt.show()

# making new features by mapping the features into polynomial terms
def mapFeatures(X1, X2):
    degrees = 6
    out = np.ones(X.shape[0])[:,np.newaxis]
    for i in range(1, degrees+1):
        for j in range(i+1):
            out = np.hstack((out, np.multiply(np.power(X1, i-j), np.power(X2, j))[:,np.newaxis]))
    return out
    
X = mapFeatures(X.iloc[:,0], X.iloc[:,1])

def sigmoid(x):
  return 1/(1+np.exp(-x))
  
def costFunctionRegression(theta, X, y, lambda_t):
    costSample = np.multiply(y, np.log(sigmoid(np.dot(X, theta))))+ np.multiply((1-y),np.log(1-sigmoid(np.dot(X, theta))))
    totalCost = (-1/m)*np.sum(costSample)
    reg = (lambda_t/(2*m))*(np.dot(theta[1:].T, theta[1:]))
    J = totalCost + reg
    return J
    
def gradient(theta, X, y, lambda_t):
    m = len(y)
    grad = np.zeros([m,1])
    temp = sigmoid(np.dot(X, theta)) - y
    grad = (1/m)*np.dot(X.T, temp)
    grad[1:] = grad[1:] + (lambda_t/m)*theta[1:]
    return grad

# initiaszing parameters
(m, n) = X.shape
y = y[:, np.newaxis]
theta = np.zeros((n,1))
lambda_t = 1

J = costFunctionRegression(theta, X, y, lambda_t)
print('cost before optimization:', J)

temp = opt.fmin_tnc(func = costFunctionRegression,
                    x0 = theta.flatten(), fprime = gradient,
                    args = (X, y.flatten(), lambda_t))

theta_optimized = temp[0]
print('optimized parameters:', theta_optimized)

J = costFunctionRegression(theta_optimized, X, y, lambda_t)
print('cost after optimization:', J)

#accuracy
pred = [sigmoid(np.dot(X, theta)) >= 0.5]
acc = np.mean(pred == y.flatten()) * 100
print(acc)


