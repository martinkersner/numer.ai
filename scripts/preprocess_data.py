#!/usr/bin/env python

'''
Martin Kersner, m.kersner@gmail.com
2016/07/24

This script prepares data for training. It should be employed only once for each dataset.

TODO
* print to STDERR
'''

import os
import sys

from settings import *
from tools import *

def main():
  if len(sys.argv) == 2:
    data_id = sys.argv[1]
    settings["data_id"] = data_id

    data_path = settings["data_path_orig"]
    train_csv = settings["train_csv_orig"]
    path = data_path.format(data_id, train_csv)

    if os.path.exists(path):
      load_orig_data(path, settings["to_save"])
    else:
      print "Dataset does not exists."

  else: 
    print "You have to specify name of dataset."

if __name__ == '__main__':
  main()
