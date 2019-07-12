# -*- coding: utf-8 -*-
# @Time    : 2019/7/11 下午7:42
# @Author  : Ryan
# @File    : 2-basic_operations.py

import tensorflow as tf

# create constant op
a = tf.constant(2)
b = tf.constant(3)

# launch the default graph
with tf.Session() as sess:
    print('a=2,b=3')
    print('常量相加: %i' % sess.run(a+b))

# create operations with variable as graph input
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

# Define some operation
add = tf.add(a,b)
mul = tf.multiply(a,b)

# launch the default graph
with tf.Session() as sess:
    print('变量相加: %i ' % sess.run(add, feed_dict={a:2, b:3}))
    print('变量相乘: %i ' % sess.run(mul, feed_dict={a:2, b:3}))

