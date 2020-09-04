import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
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

# TeX fonts
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Times']})

class GSClassification():
    '''
    Executes grid search and cross-validation for many classification models.
    
    Parameters: 
                models: list of potential classifiers
                grid: grid search parameters
                
    '''
    def __init__(self, models, grid):
        self.models = models
        
        # instances only desired models.
        self.grid_of_params = {k:v for k, v in grid.items() if k in self.models}
        
    def apply_grid_search(self, X_train, y_train, k=5):
        self.X_train = X_train
        self.y_train = y_train
        '''
        Parameters: 
                    X_train: 2D ndarray
                    y_train: 1D ndarray                
                    k: cross-validation k-fold. Default: 5.
        '''
        
        # list of current compatible classifiers
        compatible_classes = [SVC(), DecisionTreeClassifier(), KNeighborsClassifier(), LogisticRegression(), GaussianNB(), RandomForestClassifier(), SGDClassifier(), Perceptron()]

        compatible_classes_str = [str(i) for i in compatible_classes if str(i) in self.grid_of_params.keys()]

        self.classificators = [compatible_classes[i].fit(X_train, y_train) for i in range(len(compatible_classes)) if str(compatible_classes[i]) in self.grid_of_params.keys()]
        
        self.model_name = []
        self.accuracies = []
        self.standar_dev = []
        self.best_parameters = []
        self.best_estimators = []
        for i in range(len(self.classificators)):
            start = time.time()
            print("Executing grid search for {}.".format(compatible_classes_str[i]))
            grid_search = GridSearchCV(estimator = self.classificators[i],
                                    param_grid = self.grid_of_params[compatible_classes_str[i]],
                                    scoring = 'accuracy',
                                    cv = k,
                                    n_jobs = -1,
                                    verbose=1)
            grid_search.fit(X_train, y_train)
            self.accuracies.append(grid_search.best_score_)
            self.best_parameters.append(grid_search.best_params_)
            self.best_estimators.append(grid_search.best_estimator_)
            self.standar_dev.append(grid_search.cv_results_['std_test_score'][grid_search.best_index_])
            self.model_name.append(compatible_classes_str[i][0:-2])
            end = time.time()
            print ("Elapsed time: %.3fs"%(end-start))
            
        # XGboost is special...
        if 'XGBClassifier()' in self.grid_of_params.keys():
            start = time.time()
            xgb = XGBClassifier()
            print("Executing grid search for XGBClassifier().")
            grid_search = GridSearchCV(estimator = xgb,
                                    param_grid = self.grid_of_params['XGBClassifier()'],
                                    scoring = 'accuracy',
                                    cv = k,
                                    n_jobs = -1,
                                    verbose=1)
            grid_search.fit(X_train, y_train)
            self.accuracies.append(grid_search.best_score_)
            self.best_parameters.append(grid_search.best_params_)
            self.standar_dev.append(grid_search.cv_results_['std_test_score'][grid_search.best_index_])
            self.model_name.append('XGBClassifier')
            end = time.time()
            print ("Elapsed time: %.3fs"%(end-start))
            xgb.fit(X_train, y_train)
            self.classificators.append(xgb)
            self.best_estimators.append(grid_search.best_estimator_)
    
    def show_dataframe(self):
        out = list(zip(self.model_name, self.accuracies, self.standar_dev)) #zip joins same index tuples of lists
        resultsinDataFrame = pd.DataFrame(out, columns = ['method', 'mean accuracy (%)', 'standard deviation (%)'])
        final_df = resultsinDataFrame.sort_values(by='mean accuracy (%)', ascending=False)
        print(final_df)
        
    def plot_cap_curves(self):
        # split
        X_train_, X_test_, y_train_, y_test_ = train_test_split(self.X_train, self.y_train, test_size = 0.40)
        
        # used to compute CAP
        self.y_pred = []
        for best_estimator in self.best_estimators:
            self.y_pred.append(best_estimator.predict(X_test_).tolist())

        self.y_test_ = [y_test_.tolist()]*len(self.best_estimators)
        
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(7,12))
        ax1 = plt.subplot(211)
        for model in range(len(self.y_pred)):
            # sort
            data = pd.DataFrame(list(zip(self.y_test_[model],self.y_pred[model])), columns=['y','ypred'])
            # sort
            data_sorted_ypred = data.sort_values(by=['ypred'], ascending=False)
            data_sorted_y = data.sort_values(by=['y'], ascending=False)

            # total records
            total_records = len(data)
            # total amount of positives
            total_positive = len(data[data['y'] == 1])

            # proportion of the total records (x axis)
            x = [(i+1)/total_records for i in range(total_records)]

            # proportion of positives out of total
            proportion_of_positive = total_positive/total_records
            # random select
            random_select = [(i+1)*proportion_of_positive for i in range(total_records)]
            # out of the random select, proportion of positives (y axis)
            random_select_proportion_of_positive = [random_select[i]/total_positive for i in range(total_records)]

            # model select
            model_select = [sum(data_sorted_ypred.iloc[0:i+1,0]) for i in range(total_records)]
            # out of the model select, proportion of positives (y axis)
            model_select_proportion_of_positive = [model_select[i]/total_positive for i in range(total_records)]

            # perfect select
            perfect_select = [sum(data_sorted_y.iloc[0:i+1,0]) for i in range(total_records)]
            # out of the perfect select, proportion of positives (y axis)
            perfect_select_proportion_of_positive = [perfect_select[i]/total_positive for i in range(total_records)]

            auc_random = auc(x, random_select_proportion_of_positive)
            auc_model = auc(x, model_select_proportion_of_positive)
            auc_perfect = auc(x, perfect_select_proportion_of_positive)

            acc_ratio = (auc_model-auc_random)/(auc_perfect-auc_random)
            
            ax1.plot(x, model_select_proportion_of_positive, label='{}: {:.2f}'.format(self.model_name[model], acc_ratio), linewidth=0.7)
        
        ax1.plot(x, random_select_proportion_of_positive, '--', color='red', linewidth=1, label='Random', alpha=0.5)
        ax1.plot(x, perfect_select_proportion_of_positive, '--', color='blue', linewidth=1, label='Perfect Model', alpha=0.5)
        ax1.set_title('Cumulative Accuracy Profile (CAP)', size=17)
        ax1.set_xlabel('Fraction of total', fontsize=16)
        ax1.set_ylabel('Fraction of positive outcomes', fontsize=16)
        legend = ax1.legend(frameon=False, loc='lower right', title='Accuracy Ratio', fontsize=13)
        legend.get_title().set_fontsize('13')
        for legobj in legend.legendHandles:
            legobj.set_linewidth(2.0)
        plt.xticks(size=16)
        plt.yticks(size=16)
        plt.subplots_adjust(hspace=0.25)
        
        ax2 = plt.subplot(212)
        ax2.bar(self.model_name, self.accuracies, zorder=2, alpha=0.8)
        ax2.grid(alpha=0.3, zorder=0)
        ax2.errorbar(self.model_name, self.accuracies, yerr=self.standar_dev, c='C1', ls='none', zorder=3, alpha=0.8)
        ax2.set_yscale='log'
        ax2.set_title('Mean accuracy $\pm \sigma$', size=17)
        plt.xticks(rotation=10, ha='right', size=12)
        plt.yticks(size=16)
        
        
        #plt.tight_layout()
        plt.savefig('cap.jpg', dpi=150)
        plt.close()
        
    def show_best_parameters(self):
        for i in range(len(self.model_name)):
            print(self.model_name[i], self.best_parameters[i])
