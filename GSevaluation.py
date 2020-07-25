from GSClassification import GSClassification
from data_preprocessing import preprocessed_data

import numpy as np
import pandas as pd

models = ['SVC()', 
          'DecisionTreeClassifier()', 
          'KNeighborsClassifier()', 
          'LogisticRegression()', 
          'GaussianNB()', 
          'RandomForestClassifier()', 
          'Perceptron()', 
          'SGDClassifier()', 
          'XGBClassifier()'
          ]

X_train, y_train, X_to_predict, _ = preprocessed_data()

grid_searcher = GSClassification(models)

model, acc, std, best_params = grid_searcher.apply_grid_search(X_train, y_train, k=30)
for i in range(len(model)):
    print(model[i], best_params[i])

print('\n')
out = list(zip(model, acc, std)) #zip joins same index tuples of lists
resultsinDataFrame = pd.DataFrame(out, columns = ['method', 'mean accuracy (%)', 'standard deviation (%)'])
print(resultsinDataFrame.sort_values(by='mean accuracy (%)', ascending=False))
