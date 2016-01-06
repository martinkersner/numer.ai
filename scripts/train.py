#!/usr/bin/python

'''
Martin Kersner, m.kersner@gmail.com
2015/12/25
'''

import os

from tools import *
import pandas as pd

from sklearn import cross_validation as CV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

# classifiers
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import ExtraTreesClassifier as ET
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import AdaBoostClassifier as AB
from sklearn.ensemble import GradientBoostingClassifier as GB

# preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MaxAbsScaler, RobustScaler

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

from xgb import train_xgb, train_val_xgb, predict_xgb, save_model_xgb

def main():
    ## SETTINGS ################################################################
    settings = {}
    settings['verbose'] = False

    settings['id'] = '201601'
    settings['train_orig_data'] = os.path.join('../data/', settings['id'], 'orig/numerai_training_data.csv')
    settings['tour_orig_data'] = os.path.join('../data/', settings['id'], 'orig/numerai_tournament_data.csv')

    load_from_orig_data(settings['train_orig_data'], save_id=settings['id'])
    load_tournament_data(settings['tour_orig_data'], save_id=settings['id'])
    exit()

    settings['train_csv']     = os.path.join('../data/', settings['id'], 'train_v_num.csv')
    settings['val_csv']       = os.path.join('../data/', settings['id'], 'val_v_num.csv')
    
    settings['train_val_csv'] = os.path.join('../data/', settings['id'], 'train_val_v_num.csv')
    settings['test_csv']      = os.path.join('../data/', settings['id'], 'tour_v_num.csv')

    n_jobs = 4 

    settings['parameters'] = [{
        'rf__criterion': ['entropy'], 
        'rf__n_estimators': [2000],
        'rf__n_jobs': [n_jobs]
        }]

    settings['subset'] = None

    #settings['transformers']  = [Pipeline([ ('poly', PolynomialFeatures()), ('scaler', MinMaxScaler()) ])]

    settings['grid_pipeline'] = Pipeline([
        ('poly', PolynomialFeatures()),
        #('scaler', MinMaxScaler()), 
        ('rf', RF())
        ])

    ## XGB SETTINGS ############################################################
    xgb_settings = {}
    xgb_settings['num_round'] = 20
    xgb_settings['early_stop'] = 0
    xgb_params = {'bst:max_depth': 4, 
                  'bst:eta': 0.1, 
                  'bst:gamma': 1,
                  'bst:min_child_weight': 0.5,
                  'bst:max_delta_step': 0.2,
                  'silent': 1, 
                  'objective': 'binary:logistic',
                  'nthread': 4,
                  'eval_metric': 'auc'}
    xgb_settings['params'] = xgb_params
    
    #do_train_val(settings)
    #do_train_test(settings)

    #do_train_val_gs(settings)

    do_train_val_xgb(settings, xgb_settings)
    #do_train_test_xgb(settings, xgb_settings)

    #do_class_specific_train_val(settings)

    #a = pd.read_csv('../submissions/1451190098.csv')
    #b = pd.read_csv('../submissions/1451190463.csv').probability
    #a.probability = (a.probability + b) / 2.0
    #a.to_csv('../submissions/a.csv', columns = ('t_id', 'probability'), index = False)

###############################################################################
def do_train_val_xgb(settings, xgb_settings):
    train, val = load_data(settings['train_csv'], settings['val_csv'])
    y_train, X_train = split2yX(train)
    y_val, X_val     = split2yX(val)

    train_val_xgb(y_train, X_train, y_val, X_val, xgb_settings)

###############################################################################
def do_train_test_xgb(settings, xgb_settings):
    train_val, test = load_data(settings['train_val_csv'], settings['test_csv'])
    y_train_val, X_train_val = split2yX(train_val)
    id_test, X_test = test.t_id.values, test.drop('t_id', axis = 1)

    model = train_xgb(y_train_val, X_train_val, xgb_settings)
    y_pred = predict_xgb(model, X_test)
    
    uniq_id = get_timestamp_str()
    submission_path = '../submissions/' + uniq_id + '.csv'
    model_path = '../models/' + uniq_id + '.xgb_model'

    submission = pd.DataFrame({'t_id': id_test, 'probability': y_pred.T})
    submission.to_csv(submission_path, columns = ('t_id', 'probability'), index = False)
    save_model_xgb(model, model_path)

###############################################################################
def do_train_val(settings):
    train_val = load_data(settings['train_val_csv'])
    model = settings['model']
    transformers = settings['transformers']

    y_train_val, X_train_val = split2yX(train_val)

    for transformer in transformers:
        print transformer

        X_train_val_new = transformer.fit_transform(X_train_val)
        scores = CV.cross_val_score(model, X_train_val_new, y_train_val, scoring = 'roc_auc', cv = 10, verbose = 1)	

        print "%0.3f (+/-%0.03f)\n".format(scores.mean(), scores.std()*2)

###############################################################################
def do_class_specific_train_val(settings):
    col_names = get_categorical_col_names()
    #col_names = ['c1_1', 'c1_10']
    #col_names = ['c1_1']

    for col in col_names:
        print col
        settings['subset'] = col
        do_train_val_gs(settings)

###############################################################################
def do_train_val_gs(settings):
    train_val = load_data(settings['train_val_csv'])
    if settings['subset']:
        train_val = extract_categorical_subset(train_val, settings['subset'])

    #model = settings['model']
    #transformers = settings['transformers']
    #tuned_parameters = settings['tuned_parameters']
    #print tuned_parameters

    y, X = split2yX(train_val)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    clf = GridSearchCV(settings['grid_pipeline'], 
                       settings['parameters'], 
                       verbose=settings['verbose'], 
                       scoring='roc_auc', 
                       cv=5, 
                       n_jobs=1)

    clf.fit(X, y)

    print_cv_best(clf)
    #print_cv_score(clf)

    #y_true, y_pred = y_test, clf.predict(X_test)
    #auc = compute_auc(y_true, y_pred)
    #print auc

###############################################################################
def do_train_test(settings):
    train, test = load_data(settings['train_val_csv'], settings['test_csv'])
    model = settings['model']
    transformers = settings['transformers']

    y_train, X_train = split2yX(train)

    for transformer in transformers:
        print transformer

        X_train_new = transformer.fit_transform(X_train)
        X_test_new  = transformer.transform(test.drop('t_id', axis = 1))

        model.fit(X_train_new, y_train)
        p = model.predict_proba(X_test_new)
        test['probability'] = p[:,1]

        uniq_id = get_timestamp_str()
        submission_path = '../submissions/' + uniq_id + '.csv'
        model_path = '../models/' + uniq_id + '.pkl'

        test.to_csv(submission_path, columns = ('t_id', 'probability'), index = False)
        save_model(model, model_path)

if __name__ == '__main__':
    main()
