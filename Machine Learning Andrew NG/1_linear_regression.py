# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 16:57:06 2018

Linear regression using one variable and multiple variables
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def computeCost(X, y, theta):
    diff = np.dot(X, theta) - y
    cost = np.sum(np.power(diff, 2))/(2*m)
    return cost
    
def gradientDescent(X, y, alpha, theta, iterations):
    for i in range(0, iterations):
        temp = np.dot(X, theta) - y
        update_value = np.dot(X.T, temp)
        theta = theta - (alpha/m)*update_value
    return theta

data = pd.read_csv('ex1data1.txt', header = None)
X = data.iloc[:,0] #X: population
y = data.iloc[:,1] #y: profit
m = len(y)
plt.scatter(X,y)
plt.xlabel('Population in 1000s')
plt.ylabel('Profit in $10,000s')
plt.show()

#Adding the intercept term
X = X[:, np.newaxis] #to convert rank 1 arrays (m,) to rank 2 arrays (m,1)
y = y[:, np.newaxis]
ones = np.ones((m,1))
X = np.hstack((ones, X))

#initializing the parameters
theta = np.zeros([2,1])
iterations = 1500
alpha = 0.01

#computing cost
J = computeCost(X,y,theta)
print('cost before optimization:',J)


# optimizing parameters using GD
theta = gradientDescent(X, y, alpha, theta, iterations)

J = computeCost(X,y,theta)
print('cost after optimization:',J)

# Plotting the regression line
plt.scatter(X[:,1], y)
plt.xlabel('Population in 1000s')
plt.ylabel('Profit in $10,000s')
plt.plot(X[:,1], np.dot(X, theta))
plt.show()

# multivariate LR
data = pd.read_csv('ex1data2.txt', header = None)
X = data.iloc[:,0:2] # contains two variables: size of house, number of bedrooms
y = data.iloc[:,2] # house prices
m = len(y)

# feature normalization
X = (X- np.mean(X))/np.std(X)

#Adding the intercept term
y = y[:, np.newaxis]
ones = np.ones((m,1))
X = np.hstack((ones, X))

#initializing the parameters
theta = np.zeros([3,1])
iterations = 1500
alpha = 0.01

J = computeCost(X, y, theta) #cost for multivariate
print('cost before optimization:',J)
theta = gradientDescent(X, y, alpha, theta, iterations)
J = computeCost(X, y, theta) 
print('cost after optimization:',J)