#!/usr/bin/env python

'''
Martin Kersner, m.kersner@gmail.com
2016/07/24
'''

import sys
import random
import pandas as pd
from sklearn import cross_validation
import numpy as np

from tools import *

def main():
  if len(sys.argv) == 2:
    dataset_id = sys.argv[1]
    train_path = validate_train_dataset(dataset_id, orig=False)
  else:
    print "You have to specify name of dataset."

  if train_path:
    #model = create_model()

    #train_dataset = load_csv(train_path)
    #model = train(model, train_dataset)

    #test_path = validate_test_dataset(dataset_id, orig=False)
    #test_dataset = load_csv(test_path)
    #test(model, test_dataset)

    #model = train_all(model, dataset_id)

    tournament_path = validate_test_dataset(dataset_id, orig=True)
    tournament_dataset = load_csv(tournament_path)
    #predict_tournament(model, tournament_dataset)

    random_prediction(tournament_dataset)

def random_prediction(data):
  rand = []
  for i in range(0, data["t_id"].count()):
    rand.append(random.random())

  data["probability"] = rand

  # generate submission
  uniq_id = get_timestamp_str()
  submission_path = settings["submission_path"].format(uniq_id)

  data.to_csv(submission_path, columns=('t_id', 'probability'), index=False)

  print "ID: {}".format(uniq_id)

def create_model():
  model = settings['model']
  model.set_params(n_jobs=settings["cpu_num"]/2)
  model.set_params(n_estimators=100)

  return model

def train_all(model, dataset_id):
  path = validate_train_dataset(dataset_id, orig=True)
  data = load_csv(path)
  X, y = split2Xy(data)
  model.fit(X, y)

  return model

def train(model, train_data):
  #train, test = load_data(train_path, test_path)
  #transformers = settings['transformers']

  X, y = split2Xy(train_data)
  scores = cross_validation.cross_val_score(model, X, y, cv=5, scoring='log_loss', verbose=1)
  print("Cross-validation accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

  model.fit(X, y)

  #clf = model.fit(X, y_true)
  #print clf.get_params
  #print clf.feature_importances_

  #y_pred = clf.predict_proba(X)
  #print log_loss(y_true, y_pred)

  return model

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

  scores = log_loss(y_true, y_pred)
  print("Test set accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def predict_tournament(model, data):
  X = data.drop('t_id', axis=1)
  y_pred = model.predict_proba(X)
  data['probability'] = y_pred[:, 1] # TODO confirm the second column is correct

  # generate submission
  uniq_id = get_timestamp_str()
  submission_path = settings["submission_path"].format(uniq_id)
  model_path      = settings["model_path"].format(uniq_id)

  data.to_csv(submission_path, columns=('t_id', 'probability'), index=False)
  save_model(model, model_path)

  print "ID: {}".format(uniq_id)

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
