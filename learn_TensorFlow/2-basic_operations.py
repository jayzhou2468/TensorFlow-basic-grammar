# -*- coding: utf-8 -*-
# @Time    : 2019/7/12 下午5:20
# @Author  : Ryan
# @File    : 2-basic_operations.py

import tensorflow as tf

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a,b)
mul = tf.multiply(a,b)

with tf.Session() as sess:
    print('变量相加： % i' % sess.run(add, feed_dict={a:2, b:3}))
    print('变量相乘： % i' % sess.run(mul, feed_dict={a:2, b:3}))

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])

product = tf.matmul(matrix1, matrix2)

with tf.Session() as sess:
    print(sess.run(product))










