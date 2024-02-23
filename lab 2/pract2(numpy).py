# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 21:57:21 2024

@author: student
"""
# numpy

import numpy as np

a = np.array([1,2,3])
np.copy(a)
d = a.copy()
e = np.array([(1,2,3),(1,3,2)])
#print(a)

b = np.array([[(1,2,3),(1,3,2)],[(1,2,3),(2,3,7)],[(1,2,3),(2,3,7)]])
#print(b)

#print(np.linspace(2,9,0))

np.save('a', a)
c = np.load('a.npy')
#print(c)

print(b.shape)
print(b.ndim)
print(b.size)

q = a + d
print("addition of 2 array : ",q)

print("mean :",a.mean())
print("median :",np.median(e))
print("correlation of coefficient :",np.corrcoef(a))
print("standard daviation :",np.std(a))