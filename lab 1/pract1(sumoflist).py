# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 11:06:06 2024

@author: student
"""

list = [1,2,3,4,5]
print(list)
sum = 0
for i in range(len(list)):
    print(i)
    sum += list[i]
    
print('sum :',sum)