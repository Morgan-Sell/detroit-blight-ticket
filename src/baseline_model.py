import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, roc_auc_score, train_test_split
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier 

def create_train_split_baseline_dataset(dataset, val_size=0.2, random_state=3):
    '''
    Develops dataset to be used in final models.
    
    Parameters
    ----------
    dataset : df
        Dataset generated from the import_clean_train_dataset function.
    
    test_size : float
        Percentage of the data to use for the validation set.
    
    random_state : int
        Random state for train_test_split.
    
    Return
    -------
    dataset : df
        Dataset to be used for supervised ML models.
    '''
    violation_dummy = pd.get_dummies(dataset['violation_code'])
    inspector_dummy = pd.get_dummies(dataset['inspector_name'])
    dataset2 = dataset[['detroit_address', 'MI_address', 'USA_address', 'CAN_address', 'fine_amount', 'late_fee', 'compliance']].copy()
    dataset2 = pd.concat([dataset2, violation_code_dummy, inspector_dummy], axis=1)
    
    X = dataset2.drop('compliance', axis=1).values
    y = dataset2['compliance'].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=random_state)
    
    return X_train, X_val, y_train, y_val

def fit_baseline_model_produce_y_probs(X_train, X_val, y_train, y_val, max_depth=5, num_trees=20, cv=5)
    '''
    Uses RandomForest Classifier and cross validation to produce the probabilities of the x validation
    
    Parameters
    ----------
    X_train : arr
        Training set's independent training variables. 
        
    X_val : arr
        Validation set's independent variables.
    
    y_train: arr
        Training set's dependent variabless.
    
    y_val : arr
        Testing set's dependent variabless.
        
    max_depth: int
        Number of levels in each decision tree.
    
    num_trees : int
        Number of trees used in the RandomForest
    
    cv : int
        Number of folds in the K-Fold cross validation method.
        
    Return
    -------
    y_prob : arr
        The probablities that each violation in the validation set are compliant.
    
    forest_clf : slkearn.ensemble class
        The baseline model fitted to the training dataset.
    
    auc : float
        AUC score derived from y_prob and y_val.
    
    auc_cross_val : arr
        AUC scores for each run of the RandomForest. The number of values is determined by the 'cv' parameter.
    
    
    '''
    # Use the square root as the maximum features per leaf node.
    max_features = int(np.sqrt(X_train.shape[1]))
    forest_clf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, max_features=max_features)
    forest_clf.fit(X_train, y_train)
    
    auc_cross_val = cross_val_score(base_forest, X3_train, y3_train, scoring='roc_auc', cv=5)
    y_prob = forest_clf.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, y_prob)
    
    return y_prob, forest_clf, auc, auc_cross_val