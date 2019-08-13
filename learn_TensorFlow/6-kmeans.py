# -*- coding: utf-8 -*-
# @Time    : 2019/7/17 下午4:08
# @Author  : Ryan
# @File    : 6-kmeans.py

import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
full_data_x = mnist.train.images

# parameters
num_steps = 50  # total steps
batch_size = 1024
k = 25  # num of clusters
num_classes = 10
num_features = 784

# input images
X = tf.placeholder(tf.float32, shape=[None, num_features])
Y = tf.placeholder(tf.float32, shape=[None, num_classes])

# k-means parameters
kmeans = KMeans(input=X, num_clusters=k, distance_metric='cosine', use_mini_batch=True)

# build kmeans graph
training_graph = kmeans.training_graph()

if len(training_graph) > 6:
    (all_scores, cluster)


















