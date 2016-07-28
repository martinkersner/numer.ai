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
  if len(sys.argv) == 2:
    path = validate_train_dataset(sys.argv[1])
  else:
    print "You have to specify name of dataset."

  if path:
    prepare_orig_data(path, settings["to_save"])

if __name__ == '__main__':
  main()
