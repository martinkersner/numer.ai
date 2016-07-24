#!/usr/bin/env python

'''
Martin Kersner, m.kersner@gmail.com
2016/07/24

This script prepares data for training.
It should be employed only once for each dataset.
'''

import sys

from settings import *
from tools import *

def main():
  path = validate_train_dataset(sys.argv)

  if path:
    load_orig_data(path, settings["to_save"])

if __name__ == '__main__':
  main()
