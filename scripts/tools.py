#!/usr/bin/env python

'''
Martin Kersner, m.kersner@gmail.com
2016/07/24
'''

import os
import random
import pandas as pd
from sklearn.cross_validation import train_test_split

from settings import *

def validate_train_dataset(argv, orig=True):
  '''
  TODO print to STDERR
  '''
  if len(argv) == 2:
    data_id = argv[1]
    settings["data_id"] = data_id

    if orig:
      data_path = settings["data_path_orig"]
      train_csv = settings["train_csv_orig"]
    else:
      data_path = settings["data_path"]
      train_csv = settings["train_csv"]

    dataset = data_path.format(data_id, train_csv)

    if os.path.exists(dataset):
      return dataset
    else:
      print "Dataset does not exists."
      return False

  else:
    print "You have to specify name of dataset."
    return False

def load_data(*args, **kwargs):
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

def load_orig_data(file_name, to_save=False):
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
