#!/usr/bin/env python

'''
Martin Kersner, m.kersner@gmail.com
2016/07/24
'''

import random
import pandas as pd
from sklearn.cross_validation import train_test_split

from settings import *

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
