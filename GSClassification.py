import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import sys
import time

class GSClassification():
    '''
    Executes grid search and cross-validation for many classification models.
    
    Parameters: 
                models: list of classification models
                
    '''
    def __init__(self, models):
        self.models = models
        
        #------------------------------------------------ Grid of params
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
                {'objective':['binary:logistic'], 'learning_rate': [0.1, 0.2], 'max_depth': [5, 6], 'min_child_weight': [1, 2], 'subsample': [1, 0.2, 0.3, 0.7], 'colsample_bytree': [0.7, 0.5], 'n_estimators': [1000, 1200, 1500]}]}
        
        #Returns to object only wanted models.
        self.grid_of_params = {k:v for k, v in grid.items() if k in self.models}
    #------------------------------------------------
    def apply_grid_search(self, X_tr, y_tr, k=4):
        '''
        Parameters: 
                    X_train: 2D ndarray
                    y_train: 1D ndarray
                    k: cross-validation k-fold. Default: 5.
        
        Returns:
                    model name, best accuracy, standard deviation, best parameters
        '''
            
        '''
        grid_of_params is callable by model. It is a dictionary of lists of dictionaries.
        ''' 
        print("Data shape: {}".format(np.shape(X_tr)))
        list_of_classes = [SVC(), DecisionTreeClassifier(), KNeighborsClassifier(), LogisticRegression(), GaussianNB(), RandomForestClassifier(), SGDClassifier(), Perceptron()]
        list_of_classes_str = [str(i) for i in list_of_classes]

        classificator = [list_of_classes[i].fit(X_tr, y_tr) for i in range(len(list_of_classes_str)) if list_of_classes_str[i] in self.grid_of_params.keys()]
        
        model = []
        accuracies = []
        standar_dev = []
        best_parameters = []
        for i in range(len(classificator)):
            start = time.time()
            print("Executing grid search for {}.".format(list_of_classes_str[i]))
            grid_search = GridSearchCV(estimator = classificator[i],
                                    param_grid = self.grid_of_params[list_of_classes_str[i]],
                                    scoring = 'accuracy',
                                    cv = k,
                                    n_jobs = -1,
                                    verbose=1)
            grid_search.fit(X_tr, y_tr)
            accuracies.append(grid_search.best_score_)
            best_parameters.append(grid_search.best_params_)
            standar_dev.append(grid_search.cv_results_['std_test_score'][grid_search.best_index_])
            model.append(list_of_classes_str[i])
            end = time.time()
            print ("Elapsed time: %.3fs"%(end-start))
        #XGboost is special...
        if 'XGBClassifier()' in self.grid_of_params.keys():
            start = time.time()
            xgb = XGBClassifier()
            print("Executing grid search for XGBClassifier().")
            grid_search = GridSearchCV(estimator = xgb,
                                    param_grid = self.grid_of_params['XGBClassifier()'],
                                    scoring = 'accuracy',
                                    cv = k,
                                    n_jobs = -1,
                                    verbose=2)
            grid_search.fit(X_tr, y_tr)
            accuracies.append(grid_search.best_score_)
            best_parameters.append(grid_search.best_params_)
            standar_dev.append(grid_search.cv_results_['std_test_score'][grid_search.best_index_])
            model.append('XGBClassifier()')
            end = time.time()
            print ("Elapsed time: %.3fs"%(end-start))
        return model, accuracies, standar_dev, best_parameters
