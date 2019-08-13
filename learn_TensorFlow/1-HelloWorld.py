# -*- coding: utf-8 -*-
# @Time    : 2019/7/12 下午5:14
# @Author  : Ryan
# @File    : 1-HelloWorld.py

import tensorflow as tf

hello = tf.constant('Hello World!')

with tf.Session() as sess:
    print(sess.run(hello))