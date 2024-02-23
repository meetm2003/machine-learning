# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 21:46:18 2024

@author: student
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

X = [0, 6, 11, 14, 22, 5] 
Y = [1, 7, 12, 15, 21, 15] 

def best_fit(X,Y) :
    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X)
    
    numer = sum([xi*yi for xi, yi in zip(X,Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2
    
    b = numer / denum
    a = ybar - b * xbar
    
    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))
    return a, b 

a, b = best_fit(X, Y) 
# y = b0 - b1*x 
plt.scatter(X, Y) 
yfit = [a + b * xi for xi in X] 
plt.plot(X, yfit) 
plt.show()