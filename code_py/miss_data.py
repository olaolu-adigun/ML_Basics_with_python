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


#####Categorical data sorting things

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
#print(X)
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
print(X)


# Convert the nominal features into binary features
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features = [2])
print(ohe.fit_transform(X).toarray())
print(df.tail())
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


# Plot
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan','magenta', 'yellow', 'black','pink', 'lightgreen', 'lightblue','gray', 'indigo', 'orange']
weights, params = [], []
for c in np.arange(-4, 6):
	lr = LogisticRegression(penalty='l1', C=10**c, random_state=0) 
	lr.fit(X_train_std, y_train)
	weights.append(lr.coef_[1])
	params.append(10**c)

weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
	plt.plot(params, weights[:, column], label=df_wine.columns[column+1],color=color)

plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center', bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)

#plt.show()



################################
# SBS Backward Selection Algorithm for Features

from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

class SBS():

	def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
		self.scoring = scoring
		self.estimator = clone(estimator)
		self.k_features = k_features
		self.test_size = test_size
		self.random_state = random_state

		
	def fit(self, X, y):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
		dim = X_train.shape[1]
		self.indices_ = tuple(range(dim))
		self.subsets_ = [self.indices_]
		score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
		self.scores_ = [score] 
		while dim > self.k_features:
			scores = []
			subsets = []
		
		for p in combinations(self.indices_, r=dim-1):
			score = self._calc_score(X_train, y_train, X_test, y_test, p)
			scores.append(score)
			subsets.append(p)	
			best = np.argmax(scores)
			self.indices_ = subsets[best]
			self.subsets_.append(self.indices_)
			dim -= 1
			self.scores_.append(scores[best])
			self.k_score_ = self.scores_[-1]

		return self
	
	
	def transform(self, X):
		return X[:, self.indices_]
	
	def _calc_score(self, X_train, y_train, X_test, y_test, indices):
		self.estimator.fit(X_train[:, indices], y_train)
		y_pred = self.estimator.predict(X_test[:, indices])
		score = self.scoring(y_test, y_pred)
		return score
		
		
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
knn = KNeighborsClassifier(n_neighbors=2)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)