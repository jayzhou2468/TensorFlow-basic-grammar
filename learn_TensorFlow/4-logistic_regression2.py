# -*- coding: utf-8 -*-
# @Time    : 2019/7/16 下午2:27
# @Author  : Ryan
# @File    : 4-logistic_regression2.py

import tensorflow as tf

# import mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# parameters
learning_rate = 0.6
training_epochs = 25
batch_size = 100
display_step = 1

# tf graph input
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b)

# cross entry
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# initialize
init = tf.global_variables_initializer()

# start training
with tf.Session() as sess:

    # run
    sess.run(init)

    # training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)

        # loop over all batches
        for i in range(total_batch): # 一个batch的开始迭代！
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # run optimization op and cost op
            # 开始计算optimizer和cost，真正的计算正是从这里开始的！因为优化得到的结果我们无所谓所以用_表示，c代表cost
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # compute average loss
            avg_cost += c / total_batch

        # display logs per epoch step
        if (epoch+1) % display_step  == 0:
            print("epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("finish!")

    # test model #利用当前学得的参数,y是预期输出和实际预测输出对比，得到一个true or false矩阵代表本轮预测的正确与否
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

    # calculate accuracy #进行精确度的判断，tf.cast就指的是类型转换函数，reduce_mean就是求出这一个batch的平均
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # testify accuracy
    print("accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


























