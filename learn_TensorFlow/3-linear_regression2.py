# -*- coding: utf-8 -*-
# @Time    : 2019/7/15 下午4:40
# @Author  : Ryan
# @File    : 3-linear_regression2.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
rng = np.random

# parameters
learning_rate = 0.01
traing_epoch = 1000
display_step = 50

# training data
train_X = numpy.array([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                         7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
train_Y = numpy.array([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                         2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])

n_samples = train_X.shape([0])

# tf graph input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# set model weight
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# consruct a linear model
pred = tf.add(tf.multiply(X, W), b)

# 损失函数
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

# 梯度下降
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 初始化
init = tf.global_variables_initializer()

# 开始训练
with tf.Session() as sess:

    sess.run(init)

    # 拟合训练集
    for epoch in range(traing_epoch):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X:x, Y:y})

        # 展示logs每个step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print("epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c),
                  "W=", sess.run(W), "b=", sess.run(b))

    print("完成")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')




























