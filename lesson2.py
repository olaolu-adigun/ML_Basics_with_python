from pandas import DataFrame, read_csv
import pandas as pd 
import matplotlib
import numpy as np


Location = r'C:\Users\Dell\codes_py\data\weather_2012.csv'

weather_2012 = pd.read_csv(Location, parse_dates=True, index_col='Date/Time')

weather_description = weather_2012['Weather']
is_snowing = weather_description.str.contains('Snow')
print(weather_2012['Temp (C)'])
# Not super useful
print(is_snowing[:5])
print(is_snowing.astype(float)[:11])
print(is_snowing.astype(float).resample('M', how=np.mean))
print(is_snowing)
temperature = weather_2012['Temp (C)'].resample('M', how=np.median)
print(temperature)
temperature = weather_2012['Temp (C)'].resample('M', how=np.median)
is_snowing = weather_2012['Weather'].str.contains('Snow')
snowiness = is_snowing.astype(float).resample('M', how=np.mean)

# Name the columns
temperature.name = "Temperature"
snowiness.name = "Snowiness"
stats = pd.concat([temperature,snowiness], axis = 1)
print(stats)
stats.plot(kind='bar', subplots=True, figsize=(15, 10))
