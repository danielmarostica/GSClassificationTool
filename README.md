#GSClassification Tool

### data_preprocessing.py
A template to import, treat data, and export in the correct format.

### GSClassification.py
The Class itself, which needs to be tuned with the desired grid search parameters.

### model_evaluation.py
The script which joins data_preprocessing and GSClassification, returning stats about each of the evaluated models and their best parameters.

##Example
This software comes with an example from [Kaggle's introductory competition](https://www.kaggle.com/c/titanic). It will preprocess data, test different parameters for different models (executing k-fold grid searches to avoid overfitting) and return a sorted pandas dataframe with the best accuracies and parameters.

In order to modify parameters of the grid search, check sklearn's [GridSearchCV guide](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).
