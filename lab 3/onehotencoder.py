# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 22:45:22 2024

@author: student
"""
from sklearn import preprocessing

encoder = preprocessing.OneHotEncoder() 
encoder.fit([[0, 2, 1, 12], [1, 3, 5, 3], [2, 3, 2, 12], [1, 2, 4, 3]]) 
encoded_vector = encoder.transform([[2, 3, 5, 3]]).toarray() 
print("\nEncoded vector =", encoded_vector)