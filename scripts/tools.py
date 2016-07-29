#!/usr/bin/env python

'''
Martin Kersner, m.kersner@gmail.com
2016/07/24
'''

import os
from time import time
import random
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib # TOOD is it the best way to store model?
import sklearn.metrics

from settings import *

def validate_train_dataset(dataset_id, orig=True):
  return validate_dataset(dataset_id, orig=orig, train=True)

def validate_test_dataset(dataset_id, orig=True):
  return validate_dataset(dataset_id, orig=orig, test=True)

def validate_dataset(dataset_id, orig=True, train=False, test=False):
  '''
  TODO print to STDERR
  TODO train dataset should always has target feature
  '''
  if not train and not test:
    print "Warning: Both train and test flags were not specified. Employ train flag."
    train = True

  settings["data_id"] = dataset_id

  if orig:
    data_path = settings["data_path_orig"]
    if train:
      data_csv = settings["train_csv_orig"]
    else:
      data_csv = settings["test_csv_orig"]
  else:
    data_path = settings["data_path"]
    if train:
      data_csv = settings["train_csv"]
    else:
      data_csv = settings["test_csv"]

  dataset = data_path.format(dataset_id, data_csv)

  # Check existence of dataset file
  if os.path.exists(dataset):
    return dataset
  else:
    print "Dataset does not exists."
    return False

def load_csv(*args, **kwargs):
  if len(args) > 1:
    return [pd.read_csv(data) for data in args]
  else:
    return pd.read_csv(args[0])

def split2Xy(data):
  y = data.target.values
  X = data.drop("target", axis = 1)

  return X, y

def joinXy(X, y):
  y_df = pd.DataFrame({"target": y}, index=[X.index.values])
  data = X.join(y_df)

  return data

# http://stackoverflow.com/a/2257449/1212452
def generate_id():
  return ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(N))

def prepare_orig_data(file_name, to_save=False):
  df = pd.read_csv(file_name)

  X, y = split2Xy(df)

  # separate test data from training data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=settings["test_size"], random_state=0)

  if to_save:
    train_data = joinXy(X_train, y_train)
    test_data  = joinXy(X_test, y_test)

    train_data.to_csv(settings["data_path"].format(settings["data_id"], settings["train_csv"]), index=False)
    test_data.to_csv(settings["data_path"].format(settings["data_id"], settings["test_csv"]), index=False)

  return X_train, X_test, y_train, y_test

def get_timestamp_str():
  return str(int(time()))

def save_model(model, model_path):
  joblib.dump(model, model_path)

def load_model(model_path):
  return joblib.load(model_path)

def log_loss(y_true, y_pred):
  return sklearn.metrics.log_loss(y_true, y_pred)
