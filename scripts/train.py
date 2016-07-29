#!/usr/bin/env python

'''
Martin Kersner, m.kersner@gmail.com
2016/07/24
'''

import sys
import pandas as pd
from sklearn import cross_validation as CV
import numpy as np

from tools import *

def main():
  if len(sys.argv) == 2:
    train_path = validate_train_dataset(sys.argv[1], orig=False)
  else:
    print "You have to specify name of dataset."

  if train_path:
    train_dataset = load_csv(train_path)
    model = train(train_dataset)

    test_path = validate_test_dataset(sys.argv[1], orig=False)
    test_dataset = load_csv(test_path)

    test(model, test_dataset)

def train(train_data):
  #train, test = load_data(train_path, test_path)
  #transformers = settings['transformers']

  model = settings['model']
  model.set_params(n_jobs=2)
  model.set_params(n_estimators=1000)

  X, y_true = split2Xy(train_data)
  clf = model.fit(X, y_true)
  print clf.get_params
  #print clf.feature_importances_

  #y_pred = clf.predict_proba(X)
  #print log_loss(y_true, y_pred)

  return clf

  #for transformer in transformers:
  #  print transformer

  #  X_train_new = transformer.fit_transform(X_train)
  #  X_test_new  = transformer.transform(test.drop('t_id', axis=1))

  #  model.fit(X_train_new, y_train)
  #  p = model.predict_proba(X_test_new)
  #  test['probability'] = p[:,1] # TODO confirm the second column is correct

  #  # generate 
  #  uniq_id = get_timestamp_str()
  #  submission_path = '../submissions/' + uniq_id + '.csv'
  #  model_path = '../models/' + uniq_id + '.pkl'

  #  test.to_csv(submission_path, columns=('t_id', 'probability'), index=False)
  #  save_model(model, model_path)

def test(model, test_data):
  X, y_true = split2Xy(test_data)
  y_pred = model.predict_proba(X)

  print log_loss(y_true, y_pred)

def experiment(dataset):
  model = settings['model']
  transformers = settings['transformers']

  X, y = split2Xy(dataset)

  for transformer in transformers:
    print transformer

    #X_transformed = transformer.fit_transform(X)
    #scores = CV.cross_val_score(model, X_transformed, y, scoring='roc_auc', cv=5, verbose=1)	

    #print "{:0.3f} (+/-{:0.03f})\n".format(scores.mean(), scores.std()*2)

if __name__ == '__main__':
  main()
