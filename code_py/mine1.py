from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd 
import sys
import matplotlib

Location = r'C:\Users\Dell\codes_py\dish.csv'
df = pd.read_csv(Location)
print(df.tail())

print(df.dtypes())