#!/usr/bin/env python

'''
Martin Kersner, m.kersner@gmail.com
2015/12/25

TODO
* save parameters
* save dataset preprocessing steps
* save results (auc)
* automatic submission
'''

from tools import *
import pandas as pd

from sklearn import cross_validation as CV

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

from xgb import train_xgb, train_val_xgb, predict_xgb

from settings import *
from xgb_settings import *

def main():
    #load_from_orig_data(settings['train_orig_data'], save_id=settings['id'])
    #load_tournament_data(settings['tour_orig_data'], save_id=settings['id'])

    #do_train_val(settings)
    #do_train_test(settings)

    #do_train_val_gs(settings)

    #do_train_val_xgb(settings, xgb_settings)
    do_train_test_xgb(settings, xgb_settings)

    #do_class_specific_train_val(settings)

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

    xgb_save_all(id_test, y_pred, model)

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

        print "{:0.3f} (+/-{:0.03f})\n".format(scores.mean(), scores.std()*2)

###############################################################################
def do_class_specific_train_val(settings):
    col_names = get_categorical_col_names()
    #col_names = ['c1_1', 'c1_10']
    #col_names = ['c1_1']

    for col in col_names:
        print col
        settings['subset'] = col
        do_train_val_gs(settings)
        #do_train_val(settings)

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
