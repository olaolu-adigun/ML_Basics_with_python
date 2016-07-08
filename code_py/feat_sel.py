import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
csv_data = '''A,B,C,D
... 1,0,2,0,3,0,4,0
... 5,0,6,0,,8,0
... 0,0,11,0,12,0,'''

df = pd.read_csv(StringIO(csv_data))
#print(df)
#print(df.isnull().sum())
#print(df.dropna())
#print(df.dropna(axis=1))

# INTERPOLATION OF MISSING DATA WITH FEATURE MEAN
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
# axis = 0 means the column. Changing it to 1 means row
# Strategy could be mean, median, and most_frequent



#print(df)
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
#print(imputed_data)


# Categorical data sorting things

df = pd.DataFrame([['green','M', 10.1, 'class1'],['red','L',13.5, 'class2'],['blue', 'XL', 15.3, 'class1']])

df.columns = ['color', 'size', 'price','classlabel1']
#print(df)
size_mapping ={'XL':3, 'L':2, 'M':1}
df['size'] =df['size'].map(size_mapping)
#print(df)

import numpy as np
class_mapping = {'class1': 0, 'class2':1}
cm = {label:idx for idx, label in enumerate(np.unique(df['classlabel1']))}
#print(cm)

#print(class_mapping)
df['classlabel1'] = df['classlabel1'].map(class_mapping)

inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel1'] = df['classlabel1'].map(inv_class_mapping)

#print(df)


from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel1'].values)
#print(y)

#print(class_le.inverse_transform(y))
X = df[['color', 'size', 'price']].values

# Transform nominal encoding to numerical values 
print(X)
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])


# Convert the nominal features into binary features
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[0])
print(ohe.fit_transform(X).toarray())

df = pd.get_dummies(df[['price', 'color', 'size']])
print(df)


######START WITH PREPROCESSING

# Read in input 
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class_label', 'Alcohol','Malic_acid', 'Ash','Alcalinity_of_ash','Magnesium','Total phenols', 'Flavanoids','Nonflavanoid phenols','Proanthocyanins', 'Color intensity', 'Hue','OD280/OD315 of diluted wines', 'Proline']
print('Class_label', np.unique(df_wine['Class_label']))

print(df_wine.tail())

# Partition Trianing and Testing set
from sklearn.cross_validation import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.3, random_state=0)

# Normalize the Feature space
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm  = mms.transform(X_test)

#Standardize the Feature Space
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)


#Regularization Term
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l2', C = 1)
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))


# Finding the most important feature using random forest 
from sklearn.ensemble import RandomForestClassifier
feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
	print("%2d) %-*s %f" % (f + 1, 30, feat_labels[f], importances[indices[f]]))
	
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), importances[indices], color='lightblue', align='center')
plt.xticks(range(X_train.shape[1]), feat_labels, rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()
	
