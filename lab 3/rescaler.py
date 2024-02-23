# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 21:47:45 2024

@author: student
"""

from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler

filename = './diabetes.csv'
names = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
dataframe = read_csv(filename,names=names)
array = dataframe.values
x = array[:,0:5]
y = array[:,5]

scaler = MinMaxScaler(feature_range=(0,1))
rescaledx = scaler.fit_transform(x) # transform into range of 0 and 1
set_printoptions(precision=2) #after point values

print(rescaledx[0:5,:])