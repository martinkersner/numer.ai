#!/usr/bin/env python

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
import pickle

from xgb import save_model_xgb

def blend(*args, **kwargs):
  '''
  blend("sub1.csv", "sub2.csv", "sub3.csv", ..., "blend_sub.csv")
  Blend given submission files and create new submission.
  Output submission score is computed as mean of all given scores in submission files.

  TODO control of ID of row before blending
  '''
  submission = pd.read_csv(args[0])
  num_submissions = len(args) - 1

  for submission_file in args[1:-1]:
    tmp_probability = pd.read_csv(submission_file).probability
    submission.probability += tmp_probability

  submission.probability /= num_submissions
  submission.to_csv(args[-1], columns = ('t_id', 'probability'), index = False)

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
        pandas_df2csv(train,      path_join('../data/', save_id, 'train_v.csv'))
        pandas_df2csv(val,        path_join('../data/', save_id, 'val_v.csv'))

        pandas_df2csv(train_num,  path_join('../data/', save_id, 'train_v_num.csv'))
        pandas_df2csv(val_num,    path_join('../data/', save_id, 'val_v_num.csv'))

        pandas_df2csv(pd.concat([train_num, val_num]), path_join('../data/', save_id, 'train_val_v_num.csv'))

    return train_num, val_num 

def load_tournament_data(file_name, save_id=None):
    t = pd.read_csv(file_name)
    
    # encode the categorical variable as one-hot, drop the original column afterwards
    t_dummies = pd.get_dummies(t.c1)
    t_num = pd.concat((t.drop('c1', axis = 1), t_dummies.astype(int)), axis = 1)

    if save_id:
        pandas_df2csv(t_num, path_join('../data/', save_id, 'tour_v_num.csv'))

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
        print "{:0.3f} (+/-%{:0.03f}) for {0!r}".format(mean_score, scores.std()*2, params)

def print_cv_best(clf):
    print clf.best_params_
    print clf.best_score_

def path_join(*path):
  return os.path.join(*path)

def save_settings(obj, path):
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

def load_settings(path):
    with open(patht + '.pkl', 'rb') as f:
        return pickle.load(f)

def save_all(train_type, t_id, y_pred, model, settings):
  '''
  TODO replace constant strings with variables
  '''
  uniq_id = '-'.join((train_type, get_timestamp_str()))
  submission_path = uniq_id.join(('../submissions/', '.csv'))
  model_path      = uniq_id.join(('../models/', '.xgb_model'))
  settings_path   = '../settings/' + uniq_id
  
  submission = pd.DataFrame({'t_id': t_id, 'probability': y_pred.T})
  submission.to_csv(submission_path, columns = ('t_id', 'probability'), index = False)
  save_model_xgb(model, model_path)
  save_settings(settings, settings_path)
