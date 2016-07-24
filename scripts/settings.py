#!/usr/bin/env python

'''
Martin Kersner, m.kersner@gmail.com
2016/07/24
'''

settings = {}
settings["data_path_orig"] = "../data/{}/orig/{}"
settings["train_csv_orig"] = "numerai_training_data.csv"
settings["test_csv_orig"]  = "numerai_tournament_data.csv"

settings["data_path"] = "../data/{}/{}"
settings["train_csv"] = "train.csv"
settings["test_csv"]  = "test.csv"

settings["to_save"] = True

settings["test_size"] = 0.1
settings["validation_size"] = 0.1
