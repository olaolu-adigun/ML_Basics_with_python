from pandas import DataFrame, read_csv
from numpy import *
import matplotlib.pyplot as plt
import pandas as pd 
import sys
import matplotlib
import numpy as np


Location = r'C:\Users\Dell\codes_py\car.csv'
df = pd.read_csv('car.csv')
print(df.info())

X = array(zeros((1728,6)), dtype = int)
y = array(zeros((1728,4)), dtype = int16)

Buying = df['Buying'].unique()
Maintenance = df['Maintenance'].unique()
Doors = df['Doors'].unique()
Persons = df['Persons'].unique()
Lug_Boot = df['Lug_Boot'].unique()
Safety = df['Safety'].unique()

Condition = df['Condition'].unique()

print(Buying)
print(Maintenance)
print(Doors)
print(Persons)
print(Lug_Boot)
print(Safety)
print(Condition)

np.set_printoptions(threshold=np.nan)

# Encoding Buying Feature
for x in range(1,5):
	ind =(np.where(df['Buying'] == Buying[x-1]))
	X[ind,0] = 5 - x

# Encoding Maintenance Feature
for x in range(1,5):
	ind =(np.where(df['Maintenance'] == Maintenance[x-1]))
	X[ind,1] = 5 - x

# Encoidng Doors Feature
for x in range(2,6):
	ind =(np.where(df['Doors'] == Doors[x-2]))
	X[ind,2] = x

# Encoding Persons Feature
person = array([2,4,6])
for x in range(0,3):
	ind = (np.where(df['Persons'] == Persons[x]))
	X[ind,3] = person[x]

# Encoding Lug_Boot Feature
lug = array([1,2,3])
for x in range(0,3):
	ind = (np.where(df['Lug_Boot'] == Lug_Boot[x]))
	X[ind,4] = lug[x]

# Encoding Safety Feature
safe = array([1,2,3])
for x in range(0,3):
	ind = (np.where(df['Safety'] == Safety[x]))
	X[ind,5] = safe[x]
	
# Encoding Output
for x in range(0,4):
	ind = (np.where(df['Condition'] == Condition[x]))
	y[ind,x] = 1
#print(X[:,3])

print(X)
np.savetxt("feature.csv", X, delimiter=",")
np.savetxt("label.csv", y, delimiter=",")