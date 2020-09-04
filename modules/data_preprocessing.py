import numpy as np
import pandas as pd

def preprocessed_data():
    # importing the dataset
    training_set = pd.read_csv('dataset/training_set.csv')
    test_set = pd.read_csv('dataset/test_set.csv')

    dataset = [training_set, test_set]

    for data in dataset:
        # fill Nan with most common occurence
        most_popular_port = data['Embarked'].dropna().mode()[0]
        data['Embarked'] = data['Embarked'].fillna(most_popular_port)
        # encode Sex
        data['Sex'] = data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
        # encode Class (use this or OneHotEncoder)
        data['Embarked'] = data['Embarked'].map( {'Q': 0, 'S': 2, 'C': 1} ).astype(int)
        # create family size
        data['FamilySize'] = data['SibSp'] + data['Parch'] + 1 #+1 means the person themself
        # create IsAlone and fill with 1 if has family
        data['IsAlone'] = 0
        data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1
        # drop unwanted features
        data.drop(['Ticket', 'Cabin', 'Name', 'FamilySize', 'Parch', 'SibSp'], axis=1, inplace=True) # 'inplace' asks drop to mutate data
        # fill wrong Fare with most common value
        fares = data['Fare'].dropna().median()
        data['Fare'].fillna(fares, inplace=True)
        # fill null Age with most common value
        ages = data['Age'].dropna().median()
        data['Age'].fillna(ages, inplace=True)
        # discretize age and fare
        data['Age'] = pd.qcut(data['Age'], q=4, labels=(0,1,2,3))
        data['Fare'] = pd.qcut(data['Fare'], q=4, labels=(0,1,2,3))

    X_train = dataset[0].iloc[:, 2:].values
    y_train = dataset[0].iloc[:, 1].values

    # values to predict
    X_to_predict = dataset[1].iloc[:, 1:].values
    passenger_ids = dataset[1].iloc[:, 0].values

    return X_train, y_train, X_to_predict, passenger_ids

if __name__ == "__main__":
    preprocessed_data()
    
