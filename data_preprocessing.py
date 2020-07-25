# Importing the libraries
import numpy as np
import pandas as pd
import random
import pdb

#random.seed(63631)

#BEGIN Analysis of the training set----------------------------
# Importing the dataset
training_set = pd.read_csv('training_set.csv')
test_set = pd.read_csv('test_set.csv')

#training_set.info()
#print(training_set.columns.values)
#print(training_set.tail())
#print(training_set.describe())
#print(training_set.nunique()) #describes by PassengerID
#print(test_set.nunique()) #describes by PassengerID
#print(training_set[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
#print(training_set[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
#print(training_set[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
#print(training_set[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))

#Histogram
#g = sns.FacetGrid(training_set, col='Survived')
#g.map(plt.hist, 'Age', bins=20)
#plt.show()

#Grid of Pclas x Survival
#grid = sns.FacetGrid(training_set, col='Survived', row='Pclass', size=2.2, aspect=1.6)
#grid.map(plt.hist, 'Age', alpha=.5, bins=20)
#grid.add_legend()
#plt.show()

#grid = sns.FacetGrid(training_set, row='Embarked', size=2.2, aspect=1.6)
#grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
#grid.add_legend()
#plt.show()
#END Analysis -----------------------------------------------

#Defines whole_set
whole_set = [training_set, test_set]

for dataset in whole_set:
    #Fill Nan with most common occurence
    most_popular_port = dataset['Embarked'].dropna().mode()[0]
    dataset['Embarked'] = dataset['Embarked'].fillna(most_popular_port)
    #Encode Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    #Encode Class (use this or OneHotEncoder)
    dataset['Embarked'] = dataset['Embarked'].map( {'Q': 0, 'S': 2, 'C': 1} ).astype(int)
    #Create family size
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1 #+1 means the person themself
    #Create IsAlone and fill with 1 if has family
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    #Drop unwanted features
    dataset.drop(['Ticket', 'Cabin', 'Name', 'FamilySize', 'Parch', 'SibSp'], axis=1, inplace=True) #inplaces asks drop to mutate dataset
    #Fill wrong Fare with most common value
    fares = dataset['Fare'].dropna().median()
    dataset['Fare'].fillna(fares, inplace=True)
    #Fill null Age with most common value
    ages = dataset['Age'].dropna().median()
    dataset['Age'].fillna(ages, inplace=True)
    #Discretize age and fare
    dataset['Age'] = pd.qcut(dataset['Age'], q=4, labels=(0,1,2,3))
    dataset['Fare'] = pd.qcut(dataset['Fare'], q=4, labels=(0,1,2,3))


X_train = whole_set[0].iloc[:, 2:].values
y_train = whole_set[0].iloc[:, 1].values

print("X_train after pandas...")
print(whole_set[0])
#whole_set[0].to_csv('X_train_log.csv')
#whole_set[1].to_csv('y_train_log.csv')

# Values to predict
X_to_predict = whole_set[1].iloc[:, 1:].values
passenger_ids = whole_set[1].iloc[:, 0].values

#from sklearn.compose import ColumnTransformer
#from sklearn.preprocessing import OneHotEncoder
#ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
##print(X_train)
#X_train = np.array(ct.fit_transform(X_train))
#X_to_predict = np.array(ct.transform(X_to_predict))

#ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [4])], remainder='passthrough')
#X_train = np.array(ct.fit_transform(X_train))
#X_to_predict = np.array(ct.transform(X_to_predict))

#ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [8])], remainder='passthrough')
#X_train = np.array(ct.fit_transform(X_train))
#X_to_predict = np.array(ct.transform(X_to_predict))

#ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [12])], remainder='passthrough')
#X_train = np.array(ct.fit_transform(X_train))
#X_to_predict = np.array(ct.transform(X_to_predict))

#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train[:, [2]] = sc.fit_transform(X_train[:, [2]])
#X_to_predict[:, [2]] = sc.transform(X_to_predict[:, [2]])

#sc = StandardScaler()
#X_train[:, [3]] = sc.fit_transform(X_train[:, [3]])
#X_to_predict[:, [3]] = sc.transform(X_to_predict[:, [3]])

print("X_train after scaling and encoding.")
print(X_train)
print(np.shape(X_train))
def preprocessed_data():
    return X_train, y_train, X_to_predict, passenger_ids
