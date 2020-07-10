import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier 


def plot_permutation_importance(model, X, y, feat_names, n_repeats=10):
    perm_imp = permutation_importance(model, X, y, scoring='roc_auc', n_repeats=n_repeats, random_state=3)
    #feature_names = pd.Series(train_set.columns.drop('compliance'))
    results = perm_imp.importances_mean
    df_perm_import = pd.DataFrame()
    df_perm_import['feature_names'] = feat_names
    df_perm_import['perm_import_score'] = results
    df_perm_import.columns=df_perm_import.columns.str.strip()
    df_perm_import.sort_values('perm_import_score', inplace=True, ascending=False)
    
    plt.figure(figsize=(12,12))
    perm_import_top_25 = df_perm_import.iloc[:25, :]
    sns.barplot(x='perm_import_score', y='feature_names', data=perm_import_top_25, palette='Pastel2')
    plt.title('Permutation Importance - Validation Set - Random Forest - Baseline', fontsize=22)
    plt.tight_layout()
    plt.show();