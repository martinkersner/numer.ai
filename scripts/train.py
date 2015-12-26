#!/usr/bin/python

'''
Martin Kersner, m.kersner@gmail.com
2015/12/25
'''

from tools import *
from sklearn import cross_validation as CV
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import pandas as pd

from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MaxAbsScaler, RobustScaler

def main():
    settings = {}
    #settings['train_csv']     = '../data/train_v_num.csv'
    #settings['val_csv']       = '../data/val_v_num.csv'
    settings['train_val_csv'] = '../data/train_val_v_num.csv'
    settings['test_csv']      = '../data/tour_v_num.csv'

    settings['model']  = LR()
    settings['transformers']  = [Pipeline([ ('poly', PolynomialFeatures()), ('scaler', MinMaxScaler()) ])]
    
    #do_train_test(settings)
    do_train_val(settings)

def do_train_val(settings):
    train_val = load_data(settings['train_val_csv'])
    model = settings['model']
    transformers = settings['transformers']

    y_train_val, X_train_val = split2yX(train_val)

    for transformer in transformers:
        print transformer

        X_train_val_new = transformer.fit_transform(X_train_val)
        scores = CV.cross_val_score(model, X_train_val_new, y_train_val, scoring = 'roc_auc', cv = 10, verbose = 1)	

        print "mean AUC: {:.2%}, std: {:.2%} \n".format(scores.mean(), scores.std())

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

def blend():
    pass

if __name__ == '__main__':
    main()
