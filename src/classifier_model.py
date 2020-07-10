import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

def create_bool_col(x, val):
    if x == val:
        boolean = 1
    else:
        boolean = 0
    return boolean



def blight_optimized_model():
    '''
    Model imports and processes training.
    Selected features are the results of feature engineering/evaluation performed in the supervised_ml.ipynb and which features were used in the test dataset.
    
    Parameters:
    ----------
    
    
    Returns:
    --------
    predfa : series
        The classification predictions - i.e compliant or non-compliant - for the test dataset.
    '''
    
    
    # Your code here
    
    # Import and process train dataset.
    train = pd.read_csv('../data/train.csv', engine='python')
    train = train[train['compliance'].isnull() == False].copy()
    train['9-1-81(a)'] = train['violation_code'].apply(lambda x: create_bool_col(x, '9-1-81(a)'))
    train2 = train[['fine_amount', 'late_fee', 'discount_amount', 'compliance', '9-1-81(a)']].copy()

    train_disposition_dummy = pd.get_dummies(train['disposition'])
    train_agency_dummy = pd.get_dummies(train['agency_name'])


    train2 = pd.concat([train2, train_disposition_dummy, train_agency_dummy], axis=1)
    features = ['fine_amount', 'late_fee', 'discount_amount', 'Responsible by Admission','Responsible by Default', 
            'Responsible by Determination','Buildings, Safety Engineering & Env Department','Department of Public Works',
            'Detroit Police Department', 'compliance']
    train3 = train2[features].copy()
    
    # Train model
    
    X_train = train3.drop('compliance', axis=1).values
    y_train = train3['compliance'].values
    gb = GradientBoostingClassifier(learning_rate = 0.1, max_depth = 7, max_features=7,min_samples_leaf=1, 
                                min_samples_split= 0.4,n_estimators= 500 )
    gb.fit(X_train, y_train)
    
    # Import and process test dataset.
    test = pd.read_csv('../data/test.csv', engine='python')
    test['9-1-81(a)'] = test['violation_code'].apply(lambda x: create_bool_col(x, '9-1-81(a)'))
    test2 = test[['fine_amount', 'late_fee', 'discount_amount', '9-1-81(a)']].copy()

    test_disposition_dummy = pd.get_dummies(test['disposition'])
    test_agency_dummy = pd.get_dummies(test['agency_name'])
    test2 = pd.concat([test2, test_disposition_dummy, test_agency_dummy], axis=1)

    X_test = test2[['fine_amount', 'late_fee', 'discount_amount', 'Responsible by Admission','Responsible by Default', 
            'Responsible by Determination','Buildings, Safety Engineering & Env Department','Department of Public Works',
            'Detroit Police Department']].copy()
    
    # Predict probabilities and create dataframe.
    y_probs = gb.predict_proba(X_test.values)[:, 1]
    pred = pd.Series(y_probs, index=test['ticket_id'])
    
    return predfa