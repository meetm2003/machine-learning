# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 10:17:48 2024

@author: student
"""

data = 'hello world'
print('index 2 :',data[2])
print('Length of string :',len(data),', data :',data)
print(data)

data = 324.5
print('data override :',data)

# multiple assignment
a,b,c,d = 1,4,5,None
print('multiple assignment :',a,b,c,d)

if data == 123 :
    print('true\n')
elif data == 124 :
    print('false\n')
else :
    print('None\n')
    
var = 3
#for loop
print('for loop :')
for i in range(var) :
    print(i)
    
#while loop
print('\nwhile loop :')
while var < 7:
    print(var)
    var += 1
    
#tuple
print('\nTuple :')
a = (1,3,4)
print(a)

#List
print('\nList :')
myList = [1,2,3,4]
print('index',myList[2])
myList.append(5)
print(myList)
print('length :',len(myList))
for val in myList:
    print(val)

#dictionary
print('\ndictionary :')
mydict = {'a': 1, 'b': 4}
print(mydict['a'])
print(mydict)
mydict['b'] = 2
print('keys :',mydict.keys());
print('values :',mydict.values());
for i in mydict.keys():
    print(mydict[i])
    
#function
print('\nfunction :')
def mysum(i, j):
    print(i,'+',j);
    return i + j

val1 = int(input('Enter the number:'))
val2 = int(input('Enter the another number:'))

sum = mysum(val1,val2)
print(sum)        