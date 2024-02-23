# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 22:36:24 2024

@author: student
"""

from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import Binarizer

filename = './diabetes.csv'
names = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
dataframe = read_csv(filename,names=names)
array = dataframe.values
x = array[:,0:5]
y = array[:,5]

binarizer = Binarizer(threshold=0.0).fit(x)
binaryX = binarizer.transform(x)
set_printoptions(precision=2) #after point values

print(binaryX[0:5,:])