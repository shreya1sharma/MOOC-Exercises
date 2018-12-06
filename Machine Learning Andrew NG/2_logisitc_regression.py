# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 09:13:05 2018

@author: 0000016446351
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt 

def sigmoid(x):
    return 1/(1+np.exp(-x))
    
def costFunction(theta, X, y):
    costSample = np.multiply(y,np.log(sigmoid(np.dot(X, theta))))+ np.multiply((1-y),np.log(1-sigmoid(np.dot(X, theta))))
    totalCost = (-1/m)*np.sum(costSample)
    return totalCost

def gradientDescent(theta, X, y, alpha, iterations):
    for i in range(0, iterations):
       temp = sigmoid(np.dot(X, theta)) - y
       update_value = np.dot(X.T, temp)
       theta = theta  - (alpha/m)*update_value  
    return theta

def gradient(theta, X, y):
    temp = sigmoid(np.dot(X, theta)) - y
    gradient = (1/m)*np.dot(X.T, temp)
    return gradient
    
data = pd.read_csv('C:/Users/0000016446351/Desktop/ML_assi/data/ex2data1.txt', header = None)
X = data.iloc[:, 0:2]
y = data.iloc[:, 2]

# data visualization
mask = y==1
adm = plt.scatter(X[mask][0].values, X[mask][1].values, c='R' )
not_adm = plt.scatter(X[~mask][0].values, X[~mask][1].values)
plt.xlabel('exam 1 score')
plt.ylabel('exam 2 score')
plt.legend((adm, not_adm), ('Admitted', 'Rejected'))
plt.show()

#iniitalizing parameters
(m,n) = X.shape
X = np.hstack((np.ones((m,1)), X))
y = y[:, np.newaxis]
theta = np.zeros(((n+1),1))
iterations = 15000
alpha = 0.001

#computing cost   
J = costFunction(theta, X, y)
print('cost before optimization:', J)
#theta = gradientDescent(theta, X, y, alpha, iterations)


# optimizing parameters using inbuilt function fmin_tnc

#fmin_tnc is an optimization solver that finds the minimum of an unconstrained function. 
#You will pass to fmin_tnc the following inputs:
#The initial values of the parameters we are trying to optimize.
#A function that, when given the training set and a particular theta, computes the logistic regression cost and gradient with respect to theta for the dataset (X, y).
temp = opt.fmin_tnc(func = costFunction,
                    x0 = theta.flatten(), fprime = gradient,
                    args = (X, y.flatten()))
'''
Note
matrix (rank 2)(m,) -> vector (rank 1)(m,1)=> y[:, np.newaxis]
vector -> matrix => y.flatten()
'''
theta_optimized = temp[0]
print('optimized parameters:', theta_optimized)

J = costFunction(theta_optimized[:, np.newaxis], X, y)
print('cost after optimization:', J)

# plotting the decision boundary
plot_x = [np.min(X[:,1]-2), np.max(X[:,2]+2)]
plot_y = -1/theta_optimized[2]*(theta_optimized[0] 
          + np.dot(theta_optimized[1],plot_x))  


mask = y.flatten() == 1
adm = plt.scatter(X[mask][:,1], X[mask][:,2], c='R' )
not_adm = plt.scatter(X[~mask][:,1], X[~mask][:,2])
decision_boun = plt.plot(plot_x, plot_y)
plt.xlabel('exam 1 score')
plt.ylabel('exam 2 score')
plt.legend((adm, not_adm), ('Admitted', 'Rejected'))
plt.show()





