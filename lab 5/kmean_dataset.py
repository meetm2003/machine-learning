# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 23:40:37 2024

@author: student
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans 

df = pd.read_csv('iris.csv') 
df.columns = ['X1', 'X2', 'X3', 'X4', 'Y'] 
df = df.drop(['X4', 'X3'], axis=1) 

kmeans = KMeans(n_clusters=3) 
X = df.values[:, 0:2] 
kmeans.fit(X) 

df['Pred'] = kmeans.predict(X) 

sns.set_context('notebook', font_scale=1.1) 
sns.set_style('ticks') 
sns.scatterplot(x='X1', y='X2', hue='Pred', data=df)
plt.show()
