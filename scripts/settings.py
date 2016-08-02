#!/usr/bin/env python

'''
Martin Kersner, m.kersner@gmail.com
2016/07/24
'''

# Preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

# Classifiers
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import ExtraTreesClassifier as ET
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import AdaBoostClassifier as AB
from sklearn.ensemble import GradientBoostingClassifier as GB

from multiprocessing import cpu_count

settings = {}
settings["data_path_orig"] = "../data/{}/orig/{}"
settings["train_csv_orig"] = "numerai_training_data.csv"
settings["test_csv_orig"]  = "numerai_tournament_data.csv"

settings["data_path"] = "../data/{}/{}"
settings["train_csv"] = "train.csv"
settings["test_csv"]  = "test.csv"

settings["submission_path"] = '../submissions/{}.csv'
settings["model_path"] = '../models/{}.pkl'

settings["to_save"] = True

settings["test_size"] = 0.1
settings["validation_size"] = 0.1

# Training settings
settings['model'] = RF()
settings['transformers']  = [Pipeline([ ('poly', PolynomialFeatures()), ('scaler', MinMaxScaler()) ])]

settings["cpu_num"] = cpu_count()
