# -*- coding: utf-8 -*-
# @Time    : 2019/7/23 下午2:15
# @Author  : Ryan
# @File    : LSTM.py

import numpy as np
import tensorflow as tf



# 一、数据清洗
with open('./data/train_reviews.txt', 'r') as f:
    reviews = f.read()
with open('./data/train_labels.txt', 'r') as f:
    labels = f.read()

# 二、使用embedding层,将单词编码成整数(去除标点符号)
from string import punctuation

all_text = ''.join([c for c in reviews if c not in punctuation])
reviews = all_text.split('\n')
all_text = ''.join(reviews)
words = all_text.split()     # 将文本拆分为单独的单词列表

# print(all_text[:2000])

from collections import Counter

counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)    # 按计数进行排序
vocab_to_int = {word: i for i, word in enumerate(vocab, 1)}     # 生成字典：{单词：整数} # enumerate用于将可遍历的数据对象组合为一个索引

# 将文本列表转换为整数列表
reviews_ints = []
# for each in reviews:
#     print(each.split())
#     print([vocab_to_int[word] for word in each.split()])
#
#     exit()
for each in reviews:
    reviews_ints.append([vocab_to_int.get(word, 201) for word in each.split()])
# for i in reviews_ints:
#     print(i)
print("评论总个数： {}".format(len(reviews_ints)))

# print(len(reviews_ints))
# print(reviews_ints[0])

# # 移除长度为0的评论
# non_zero_idx = [i for i, review in enumerate(reviews_ints) if len(review)>0]
# labels = [labels[i] for i in non_zero_idx]

# 对标签进行编码，将标签转为0/1,统计句子长度
# labels = np.array([0 if label == 'negtive' else 1 for label in labels.split()])
review_lens = Counter([len(x) for x in reviews_ints])
print("长度为0的评论： {}".format(review_lens[0]))
print("最长的评论： {}".format(max(review_lens)))


# # 统计句子长度
# review_lens = Counter([len(x) for x in reviews_ints])
# print("长度为0的评论个数: {}".format(review_lens[0]))
# print("最长评论长度: {}".format(max(review_lens)))

# 去掉长度为0的评论
revice_len_zero = 0

for i, review in enumerate(reviews_ints, 0):
    if len(review) == 0:
        revice_len_zero = i
print(revice_len_zero)

reviews_ints = [review_int for review_int in reviews_ints if len(review_int) > 0]

# # 全部变为200词,keras方式
# from keras.preprocessing.sequence import pad_sequences

# features = np.array(reviews_ints)
# features = pad_sequences(features, maxlen=200, padding='post',)   # value=222
#
# print(features.shape)

# 全部变为200词
seq_len = 200
reviews_ints = [review[:200] for review in reviews_ints]
features = []
for review in reviews_ints:
    if len(review) < seq_len:
        s = []
        for i in range(seq_len - len(review)):
            s.append(0)
        s.extend(review)    # extend与append区别
        features.append(s)
    else:
        features.append(review)
features = np.array(features)

# print(features[:10, :100])
# print(features[-1])


# 划分训练集数据集
split_frac = 0.8
from sklearn.model_selection import train_test_split

train_x, val_x = train_test_split(features, test_size = 1 - split_frac, random_state = 0)
train_y, val_y = train_test_split(features, test_size = 1 - split_frac, random_state = 0)
val_x, test_x = train_test_split(val_x, test_size = 0.5, random_state = 0)
val_y, test_y = train_test_split(val_y, test_size = 0.5, random_state = 0)

print("Features shapes:")
print("train set: {}".format(train_x.shape),
      "train_y set: {}".format(val_x.shape),
      "test set: {}".format(test_x.shape))


# 三、构建模型图
lstm_size = 256
lstm_layers = 1
batch_size = 128
learning_rate = 0.001

n_words = len(vocab_to_int) + 1


inputs = tf.placeholder(tf.int32, [None, 200], name='inputs')
labels_ = tf.placeholder(tf.int32, [None, 1], name='labels')
keep_prod = tf.placeholder(tf.float32, name='keep_prod')

# 嵌入层：单词嵌入，代替one-hot
embed_size = 300

embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
embedded = tf.nn.embedding_lookup(embedding, inputs)

# LSTM层
lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)  # 创建LSTM细胞层
# dropout:网络中每个单元在每次有数据流入时以一定的概率(keep prob)正常工作，否则输出0值。这是是一种有效的正则化方法，可以有效防止过拟合。
drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prod=keep_prod)
cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)

initial_state = cell.zero_state(batch_size, tf.float32)

# RNN正向通过
outputs, final_state = tf.nn.dynamic_rnn(cell, embedded, initial_state=initial_state)   # 数据流入RNN节点中

# 计算输出
predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
cost = tf.losses.mean_squared_error(labels, predictions)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# 准确率
correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 数据分批,数据分成数据量相同的组
def get_batches(x, y, batch_size=100):
    n_batches = len(x)
    x, y = x[:n_batches * batch_size], y[:n_batches * batch_size]
    for i in range(0, len(x), batch_size):
        yield x[i:i+batch_size], y[i:i+batch_size]

# 开始训练
epochs = 50
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer)
    iteration = 1
    for e in range(epochs):
        state = sess.run()

        for i, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
            feed = {inputs : x,
                    labels : y[:, None],
                    keep_prod : 0.5,
                    initial_state : state}
            loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)

        if iteration % 5 == 0:
            print("Epoch: {}/{}".format(e, epochs),
                  "Iteration: {}".format(iteration),
                  "Train loss: {:.3f}".format(loss))
        if iteration % 25 == 0:
            val_acc = []
            val_state = sess.run(cell.zero_state(batch_size, tf.float32))
            for x, y in get_batches(val_x, val_y, batch_size):
                feed = {
                    inputs : x,
                    labels : y[:, None],
                    keep_prod : 1,
                    initial_state : val_state
                }
                batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                val_acc.append(batch_acc)
            print("val acc: {:.3f}".format(np.mean(val_acc)))
        iteration += 1
    saver.save(sess, 'checkpoints/sentiment.ckpt')















































































