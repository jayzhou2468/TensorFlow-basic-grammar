# -*- coding: utf-8 -*-
# @Time    : 2019/7/12 下午5:48
# @Author  : Ryan
# @File    : 3-linear_regression.py

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy
rng = numpy.random


import numpy as np
data = np.random.randint(100, size=(2, 10))
print(data)


# paramters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# training data
train_X = numpy.array([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                         7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
train_Y = numpy.array([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                         2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])


print(train_X.shape)   # shape=(17,)
n_samples = train_X.shape[0]

X = tf.placeholder('float')
Y = tf.placeholder('float')

W = tf.Variable(rng.randn(), name='weight')
b = tf.Variable(rng.randn(), name='bias')

# constract a linear model
pred = tf.add(tf.multiply(X, W), b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

# gradient desent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# initialize variables
init = tf.global_variables_initializer()

# start training
with tf.Session() as sess:

    # run initializer
    sess.run(init)

    # fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost,feed_dict={X: train_X, Y: train_Y})
            print('Epoch:', '%04d' % (epoch+1), 'cost=', '{:.9f}'.format(c),
                  'W=', sess.run(W), 'b=', sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')

    plt.legend()
    plt.show()

    # testing example
    test_X = numpy.asarray([6.83, 4.668, 8.9, 7.91, 3.1, 5.7, 8.7, 2.1])
    test_Y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    print("Testing... (均方损失比较)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})
    print("Testing cost=", testing_cost)
    print("绝对均方损失区别:", abs(training_cost - testing_cost))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()












