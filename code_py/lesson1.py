from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd 
import sys
import matplotlib


 
print('Python version ' + sys.version)
print('Pandas version ' + pd.__version__)
print('Matplotlib version ' + matplotlib.__version__)

# The inital set of baby names and bith rates
names = ['Bob','Jessica','Mary','John','Mel']
births = [968, 155, 77, 578, 973]

# Merge lists with zip
BabyDataSet = list(zip(names,births))
print(BabyDataSet)
df = pd.DataFrame(data = BabyDataSet, columns = ['Names', 'Birth'])
print(df)

# read the data into a file
#df.to_csv
df.to_csv('births1880.csv', index = False, header = False)


Location = r'C:\Users\Dell\codes_py\births1880.csv'
df = pd.read_csv(Location, names = ['Names', 'Births'])
print(df)

import os
os.remove(Location)
print(df.dtypes)
print(df.Births.dtypes)

sorted = df.sort_values(['Births'], ascending = False)
print(sorted)
print(df['Births'].max())

df['Births'].plot()


#plt.show()