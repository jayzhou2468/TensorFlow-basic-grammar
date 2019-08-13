# -*- coding: utf-8 -*-
# @Time    : 2019/7/17 下午6:31
# @Author  : Ryan
# @File    : basic_rnn.py

from utils import *
import tensorflow as tf
# import sklearn.gfgj
from sklearn.model_selection import train_test_split
import time

trainset = sklearn.datasets.load_files(container_path='data', encoding='UTF-8')
trainset.data, trainset.target = separate_dataset(trainset, 1.0)
print(trainset.target_names)
print(trainset.data)
print(trainset.target)
print(len(trainset.data))
print(len(trainset.target))