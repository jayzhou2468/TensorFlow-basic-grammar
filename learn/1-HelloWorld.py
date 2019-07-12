# -*- coding: utf-8 -*-
# @Time    : 2019/7/11 下午7:36
# @Author  : Ryan
# @File    : 1-HelloWorld.py

import tensorflow as tf

# create a Constant op
hello = tf.constant('Hello World!')

# start session
sess = tf.Session()

# run op
print(sess.run(hello))
