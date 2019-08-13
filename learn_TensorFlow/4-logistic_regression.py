# -*- coding: utf-8 -*-
# @Time    : 2019/7/15 下午7:03
# @Author  : Ryan
# @File    : 4-logistic_regression.py

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("tmp/data/", one_hot=True)

# parameters
learning_rate = 0.01
training_epochs = 50
batch_size = 100
display_step = 1

# tf Graph input
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b)   # softmax

# cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# initialize
init = tf.global_variables_initializer()

# start training
with tf.Session() as sess:
    sess.run(init)

    # training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        # loop over all loops
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # run optimization op & cost op
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})

            # compute average loss
            avg_cost += c / total_batch

        # display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("完成")

    # test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

    # calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
























