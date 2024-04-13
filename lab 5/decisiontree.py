# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 22:26:58 2024

@author: student
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import tree

df = pd.read_csv("IRIS.csv")
df.columns = ["X1", "X2", "X3", "X4", "Y"]
df.head()

#implementation 
from sklearn.model_selection import train_test_split
decision = tree.DecisionTreeClassifier(criterion="gini")
X = df.values[:, 0:4]
Y = df.values[:, 4]
trainX, testX, trainY, testY = train_test_split( X, Y, test_size = 0.3) 
decision.fit(trainX, trainY) 
print("Accuracy: \n", decision.score(testX, testY)) 

#Visualisation 
from six import StringIO  
from IPython.display import Image 
import pydotplus as pydot 
import graphviz
dot_data = StringIO() 
tree.export_graphviz(decision, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 

# Save the image to a file
graph.write_png('decision_tree.png')

