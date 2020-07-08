import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def import_clean_train_dataset():
    '''
    Imports, cleans and performs feature engineering on clean dataset.
    
    Parameters
    ----------
    
    
    Return
    -------
    dataset : df
        Dataset to be used for supervised ML models.
    '''
    
    train = pd.read_csv('../data/train.csv', engine='python')
    train2 = train[train['compliance'].isnull() == False].copy()
    
    # Feature engineering
    train2['detroit_address'] = train2['city'].apply(lambda x: create_bool_col(x, 'Detroit'))
    train2['MI_address'] = train2['state'].apply(lambda x: create_bool_col(x, 'MI'))
    train2['USA_address'] = train2['country'].apply(lambda x: create_bool_col(x, 'USA'))
    train2['CAN_address'] = train2['country'].apply(lambda x: create_bool_col(x, 'Cana'))
    
    return dataset

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


def create_train_val_split(dataset, val_size=0.2, random_state=3):
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
    
    violation_code_dummy = pd.get_dummies(train2['violation_code'])
    inspector_dummy = pd.get_dummies(train2['inspector_name'])
    disposition_dummy = pd.get_dummies(train2['disposition'])
    dataset2 = dataset[['fine_amount', 'late_fee', 'discount_amount','compliance']].copy()
    dataset2 = pd.concat([dataset2, violation_code_dummy, disposition_dummy, inspector_dummy, agency_dummy], axis=1)
    
    X = dataset2.drop('compliance', axis=1).values
    y = dataset2['compliance'].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=random_state)
    
    return X_train, X_val, y_train, y_val