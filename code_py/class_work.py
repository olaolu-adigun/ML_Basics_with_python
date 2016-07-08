import pandas as pd
import numpy as np
df2 = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header=None)
df1 = pd.read_csv('adult.csv', header = None)
df = df2.append(df1)

df.columns = ['Age','Workclass', 'Fnlwgt', 'Education', 'Education_Num', 'Marital_Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Capital-Gain', 'Capital-Loss',	'Hours-per-Week', 'Countries', 'ClassLabel']

class_map = {' <=50K':1, ' >50K':2}
df['ClassLabel'] = df['ClassLabel'].map(class_map)

y = df.iloc[:, 14].values

#Work_map = {label:idx for idx, label in enumerate(np.unique(df['Workclass']))}

is_noWork = df['Workclass']== ' Never-worked'
df.set_value(is_noWork, 'Workclass', ' Without-pay')


edu = df['Education']== ' Preschool'
df.set_value(edu, 'Education', ' Basic')
edu = df['Education']== ' 1st-4th'
df.set_value(edu, 'Education', ' Basic')
edu = df['Education']== ' 5th-6th'
df.set_value(edu, 'Education', ' Basic')

edu = df['Education']== ' 9th'
df.set_value(edu, 'Education', ' High')
edu = df['Education']== ' 10th'
df.set_value(edu, 'Education', ' High')
edu = df['Education']== ' 11th'
df.set_value(edu, 'Education', ' High')
edu = df['Education']== ' 12th'
df.set_value(edu, 'Education', ' High')


nat = df['Countries'] != ' United-States'
nat1 = df['Countries'] != ' ?'
nat2 = nat & nat1
df.set_value(nat2, 'Countries', ' International')


#count = df['Education'].value_counts()
#print(count)

df = pd.get_dummies(df[['Age','Workclass', 'Fnlwgt', 'Education', 'Education_Num', 'Marital_Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Capital-Gain', 'Capital-Loss',	'Hours-per-Week', 'Countries']])

#print(df.head())

#count = df['Education'].value_counts()
#print(count)
# print(df.tail())
#work_count = df['Workclass'].value_counts()
#print(df[is_noWork]['Workclass'])
#print('ClassLabel', np.unique(df['ClassLabel']))

#### Divide the Data
X = df.iloc[:, 0:].values

# Partition Trianing and Testing set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.2, random_state=0)

#Standardize the Feature Space
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)


#Regularization Term + Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l2', C = 0.5)
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))


# Decision Tree
from sklearn.tree import DecisionTreeClassifier
tree =  DecisionTreeClassifier(criterion ='entropy', max_depth = 7 , random_state = 0)
tree.fit(X_train_std, y_train)
print('Training accuracy:', tree.score(X_train_std, y_train))
print('Test accuracy:', tree.score(X_test_std, y_test))

# Knn
from sklearn.neighbors import KNeighborsClassifier
knn =  KNeighborsClassifier(n_neighbors = 10, p = 2, metric = 'minkowski')
knn.fit(X_train_std, y_train)
print('Training accuracy:', knn.score(X_train_std, y_train))
print('Test accuracy:', knn.score(X_test_std, y_test))