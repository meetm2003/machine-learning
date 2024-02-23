# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 22:08:42 2024

@author: student
"""

import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import datasets, linear_model 
from sklearn.metrics import mean_squared_error, r2_score 

# Load the diabetes dataset 
diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets 
diabetes_X_train = diabetes_X[:-30] 
diabetes_X_test = diabetes_X[-30:] 

# Split the targets into training/testing sets 
diabetes_y_train = diabetes.target[:-30] 
diabetes_y_test = diabetes.target[-30:] 

regr = linear_model.LinearRegression() 

regr.fit(diabetes_X_train, diabetes_y_train)

diabetes_y_pred = regr.predict(diabetes_X_test) 

print('Coefficients: \n', regr.coef_)

print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred)) 

print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

plt.scatter(diabetes_X_test, diabetes_y_test,  color='black') 
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3) 
plt.xticks(()) 
plt.yticks(()) 
plt.show() 