from pandas import DataFrame, read_csv
import pandas as pd 
import matplotlib
import numpy as np

request = pd.read_csv('311-service-requests.csv')
#print(request['Incident Zip'].unique())
na_values = ['NO CLUE', 'N/A', '0']
requests = pd.read_csv('311-service-requests.csv', na_values = na_values, dtype = {'Incident Zip': str})
print(requests['Incident Zip'].unique())

rows_with_dashes = requests['Incident Zip'].str.contains('-').fillna(False)
len(requests[rows_with_dashes])

print(requests[rows_with_dashes])