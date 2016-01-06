#!/usr/bin/python

'''
Martin Kersner, m.kersner@gmail.com
2015/12/25

inspired by https://github.com/zygmuntz/numer.ai
'''

import os
import pandas as pd
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import accuracy_score as accuracy
from time import time
from sklearn.externals import joblib

def blend(*args, **kwargs):
    pass
    #submission = pd.DataFrame({'t_id': id_test, 'probability': y_pred.T})
    #a = [pd.read_csv(data).probability for data in args]

def load_data(*args, **kwargs):
    if len(args) > 1:
        return [pd.read_csv(data) for data in args]
    else:
        return pd.read_csv(args[0])

def load_from_orig_data(file_name, save_id=None):
    d = pd.read_csv(file_name)
    
    # indices for validation examples
    iv = d.validation == 1
    
    val   = d[iv].copy()
    train = d[~iv].copy()

    assert( set(train.c1.unique()) == set(val.c1.unique()) )
    
    # no need for validation flag anymore
    train.drop('validation', axis = 1 , inplace = True)
    
    # move the target column to front
    cols = train.columns
    cols = cols.insert(0, 'target')
    cols = cols[:-1]
    
    train = train[cols]
    val   = val[cols]
    
    # encode the categorical variable as one-hot, drop the original column afterwards
    train_dummies = pd.get_dummies(train.c1)
    train_num = pd.concat((train.drop('c1', axis = 1), train_dummies.astype(int)), axis = 1)
    
    val_dummies = pd.get_dummies(val.c1)
    val_num = pd.concat((val.drop('c1', axis = 1), val_dummies.astype(int)), axis = 1)

    if save_id:
        pandas_df2csv(train,      os.path.join('../data/', save_id, 'train_v.csv'))
        pandas_df2csv(val,        os.path.join('../data/', save_id, 'val_v.csv'))

        pandas_df2csv(train_num,  os.path.join('../data/', save_id, 'train_v_num.csv'))
        pandas_df2csv(val_num,    os.path.join('../data/', save_id, 'val_v_num.csv'))

        pandas_df2csv(pd.concat([train_num, val_num]), os.path.join('../data/', save_id, 'train_val_v_num.csv'))

    return train_num, val_num 

def load_tournament_data(file_name, save_id=None):
    t = pd.read_csv(file_name)
    
    # encode the categorical variable as one-hot, drop the original column afterwards
    t_dummies = pd.get_dummies(t.c1)
    t_num = pd.concat((t.drop('c1', axis = 1), t_dummies.astype(int)), axis = 1)

    if save_id:
        pandas_df2csv(t_num, os.path.join('../data/', save_id, 'tour_v_num.csv'))

    return t_num

def pandas_df2csv(df, csv_path):
    df.to_csv(csv_path, index = False)

def train_and_evaluate(y_train, x_train, y_val, x_val, alg):
	alg.fit(x_train, y_train)

	p = alg.predict_proba(x_val)
	p_bin = alg.predict(x_val)

	acc = accuracy(y_val, p_bin)
	auc = AUC(y_val, p[:,1])
	
	return (auc, acc)
	
def transform_train_and_evaluate(y_train, x_train, y_val, x_val, transformer, alg):
	x_train_new = transformer.fit_transform(x_train)
	x_val_new   = transformer.transform(x_val)
	
	return train_and_evaluate(y_train, x_train_new, y_val, x_val_new, alg)

def split2yX(data):
    y = data.target.values
    X = data.drop('target', axis = 1)

    return y, X

def get_timestamp_str():
    return str(int(time()))

def save_model(model, model_path):
    joblib.dump(model, model_path)

def load_model(model_path):
    return joblib.load(model_path) 

def compute_auc(y_true, y_pred):
	return AUC(y_true, y_pred)

def get_categorical_col_names():
    col_names = ['c1_1',  'c1_10', 'c1_11', 'c1_12', 'c1_13', 'c1_14',
                 'c1_15', 'c1_16', 'c1_17', 'c1_18', 'c1_19', 'c1_20',
                 'c1_21', 'c1_22', 'c1_23', 'c1_24', 'c1_3',  'c1_4',
                 'c1_5',  'c1_6',  'c1_7',  'c1_8',  'c1_9']

    return col_names

def extract_categorical_subset(data, name_field):
    categorical_col_names = get_categorical_col_names()

    index = data[name_field] == 1
    subset = data[index].copy()
    subset = subset.drop(categorical_col_names, axis = 1)

    return subset

def print_cv_score(clf):
    '''
    As input expect ouput of GridSearchCV
    '''
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std()*2, params))

def print_cv_best(clf):
    print clf.best_params_
    print clf.best_score_
