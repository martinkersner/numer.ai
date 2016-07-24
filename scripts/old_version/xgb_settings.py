#!/usr/bin/env python

'''
Martin Kersner, m.kersner@gmail.com
2016/01/07
'''

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
