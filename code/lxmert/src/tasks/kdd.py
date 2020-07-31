# coding=utf-8
import collections
import csv

from tqdm import tqdm

from param import args
from tasks.kdd_model import KDD

SHUFFLE_TIME = 10

if __name__ == "__main__":
    # Build Class
    kdd = KDD()

    kdd.predict('testB', save=True)
    
