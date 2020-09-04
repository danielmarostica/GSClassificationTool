import sys
import numpy as np

sys.path.insert(1, 'modules/')

from GSClassification import GSClassification
from data_preprocessing import preprocessed_data

# choose classifiers
classifiers = [
          'SVC()', 
          #'DecisionTreeClassifier()', 
          'KNeighborsClassifier()', 
          #'LogisticRegression()', 
          #'GaussianNB()', 
          'RandomForestClassifier()', 
          'Perceptron()', 
          #'SGDClassifier()', 
          #'XGBClassifier()'
          ]

# choose grid parameters
grid = {'SVC()': [
            {'C': [0.45, 0.5, 0.55], 'kernel': ['rbf'], 'gamma': [0.15, 0.2, 0.25], 'tol': [1e-3, 2e-3, 3e-3]}],
        'DecisionTreeClassifier()': [
            {'criterion': ['gini','entropy'], 'splitter': ['random','best'], 'max_features':['auto','sqrt','log2']}],
        'KNeighborsClassifier()': [
            {'n_neighbors': [3,4,5], 'metric': ['minkowski']}],
        'LogisticRegression()': [
            {'C': [0.25, 0.5, 0.75, 1], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}],
        'GaussianNB()': [
            {'var_smoothing': [1e-11, 5e-10, 1e-9]}],
        'RandomForestClassifier()': [
            {'criterion': ['gini','entropy'], 'max_features':['auto','sqrt','log2']}],
        'SGDClassifier()': [
            {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'], 'penalty': ['l2', 'l1', 'elasticnet'], 'alpha': [0.0001, 0.001]}],
        'Perceptron()': [
            {'penalty': ['l2', 'l1', 'elasticnet']}],
        'XGBClassifier()': [
            {'objective':['binary:logistic'], 'learning_rate': [0.1, 0.2], 'max_depth': [5, 6], 'min_child_weight': [1, 2], 'subsample': [1, 0.7], 'colsample_bytree': [0.7, 0.5], 'n_estimators': [1000, 1500]}]}

# import data
X_train, y_train, _, _ = preprocessed_data()

# create an instance of the class
grid_searcher = GSClassification(classifiers, grid)

# apply grid search
grid_searcher.apply_grid_search(X_train, y_train, k=20)

# a terminal-printed dataframe with ranking and scores
grid_searcher.show_dataframe()

# print best parameters of selected classifiers
grid_searcher.show_best_parameters()

# plot cumulative accuracy profiles and AUC ratios
grid_searcher.plot_cap_curves()

