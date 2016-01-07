#!/usr/bin/env python

'''
Martin Kersner, m.kersner@gmail.com
2016/01/07
'''

# preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

# classifiers
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import ExtraTreesClassifier as ET
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import AdaBoostClassifier as AB
from sklearn.ensemble import GradientBoostingClassifier as GB

from tools import path_join

settings = {}
settings['verbose'] = False

settings['id'] = '201601'
settings['train_orig_data'] = path_join('../data/', settings['id'], 'orig/numerai_training_data.csv')
settings['tour_orig_data']  = path_join('../data/', settings['id'], 'orig/numerai_tournament_data.csv')

settings['train_csv']     = path_join('../data/', settings['id'], 'train_v_num.csv')
settings['val_csv']       = path_join('../data/', settings['id'], 'val_v_num.csv')

settings['train_val_csv'] = path_join('../data/', settings['id'], 'train_val_v_num.csv')
settings['test_csv']      = path_join('../data/', settings['id'], 'tour_v_num.csv')

n_jobs = 4 

settings['model'] = RF()

settings['parameters'] = [{
    'rf__criterion': ['entropy'], 
    'rf__n_estimators': [2000],
    'rf__n_jobs': [n_jobs]
    }]

settings['subset'] = None
settings['transformers']  = [Pipeline([ ('poly', PolynomialFeatures()), ('scaler', MinMaxScaler()) ])]

settings['grid_pipeline'] = Pipeline([
    ('poly', PolynomialFeatures()),
    #('scaler', MinMaxScaler()), 
    ('rf', RF())
    ])
