#!/usr/bin/env python

'''
Martin Kersner, m.kersner@gmail.com
2016/07/24
'''

import sys
import pandas as pd
from sklearn import cross_validation as CV

from tools import *

def main():
  #path = validate_train_dataset(sys.argv, orig=False)

  #if path:
  #  dataset = load_dataset(path)
  
  data_id = "201607"
  train_path = "../data/{}/train.csv".format(data_id)
  test_path  = "../data/{}/orig/numerai_tournament_data.csv".format(data_id)
  train_test(train_path, test_path)

def load_dataset(path):
  return pd.read_csv(path)


def train_test(train_path, test_path):
  train, test = load_data(train_path, test_path)

  model = settings['model']
  transformers = settings['transformers']

  X_train, y_train = split2Xy(train)

  for transformer in transformers:
    print transformer

    X_train_new = transformer.fit_transform(X_train)
    X_test_new  = transformer.transform(test.drop('t_id', axis=1))

    model.fit(X_train_new, y_train)
    p = model.predict_proba(X_test_new)
    test['probability'] = p[:,1] # TODO confirm the second column is correct

    # generate 
    uniq_id = get_timestamp_str()
    submission_path = '../submissions/' + uniq_id + '.csv'
    model_path = '../models/' + uniq_id + '.pkl'

    test.to_csv(submission_path, columns=('t_id', 'probability'), index=False)
    save_model(model, model_path)

def experiment(dataset):
  model = settings['model']
  transformers = settings['transformers']

  X, y = split2Xy(dataset)

  for transformer in transformers:
    print transformer

    #X_transformed = transformer.fit_transform(X)
    #scores = CV.cross_val_score(model, X_transformed, y, scoring='roc_auc', cv=5, verbose=1)	

    #print "{:0.3f} (+/-{:0.03f})\n".format(scores.mean(), scores.std()*2)

def train(dataset):
  pass

def apply_model(dataset, model):
  pass

if __name__ == '__main__':
  main()
