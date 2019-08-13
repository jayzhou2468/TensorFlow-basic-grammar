# -*- coding: utf-8 -*-
# @Time    : 2019/7/16 下午3:18
# @Author  : Ryan
# @File    : 5-nearest_neighbor.py

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# limit mnist data
Xtr, Ytr = mnist.train.next_batch(5000)  # 5000 for training
Xte, Yte = mnist.test.next_batch(200)    # 200 for testing

# tf graph input
xtr = tf.placeholder("float", [None, 784])
xte = tf.placeholder("float", [784])

# nearest neighbor calculation using L1 distance
# calculate L1 distance
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
# prediction: get min distance index (nearest neighbor)
pred = tf.arg_min(distance, 0)

accuracy = 0

# initialize the variables
init = tf.global_variables_initializer()

# start training
with tf.Session() as sess:

    sess.run(init)

    # loop
    for i in range(len(Xte)):
        # get nearest neighbor # 获取当前样本的最近邻索引
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
        # get nn class label and compare it to its true label # 最近邻分类标签与真实标签比较
        print("test", i, "prediction:", np.argmax(Ytr[nn_index]),
              "true class:", np.argmax(Yte[i]))
        # calculate accuracy  # 计算精确度
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1./len(Xte)

    print("done!")
    print("accuracy:", accuracy)




















