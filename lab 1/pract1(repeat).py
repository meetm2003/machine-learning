# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 11:15:32 2024

@author: student
"""

tup = (1,2,4,5,1)
for i in tup:
    if tup.count(i)>1:
        print('Repeated')
    else:
        print(i)