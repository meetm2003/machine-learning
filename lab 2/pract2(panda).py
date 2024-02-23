# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 22:36:06 2024

@author: student
"""

import pandas as pd

s = pd.Series([3,-5,2,1], index=['a','b','c','d'])
print(s)

data = {'Country': ['Belgium', 'India', 'Brazil'], 
 'Capital': ['Brussels', 'New Delhi', 'Brasília'],
 'Population': [11190846, 1303171035, 207847528]}

df = pd.DataFrame(data, columns=['Country', 'Capital', 'Population'])

print(df)

pd.read_csv('./data.csv')
df.to_csv('df.csv')

pd.read_excel('./data.xlsx')
