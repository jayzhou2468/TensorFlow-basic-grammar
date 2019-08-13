"""

@file  : data_helper.py

@author: xiaolu

@time  : 2019-07-22

"""
import re
from keras.utils import to_categorical  # 转为one-hot
import tensorflow as tf


def load_data(path, lab):
    # 加载数据
    data = []
    label = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            data.append(line)
            label.append(lab)
    return data, label


def clean_data(string):
    # 简单对数据进行清洗
    # 对数据清洗
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


if __name__ == '__main__':
    # 路径　　这里我们将消极标志为0 积极标志为１
    neg_path = './data/negative/negative'
    neg_sign = 0
    pos_path = './data/positive/positive'
    pos_sign = 1
    neg_data, neg_label = load_data(neg_path, neg_sign)
    pos_data, pos_label = load_data(pos_path, pos_sign)
    print("消极样本个数:", len(neg_data))
    print("积极样本个数:", len(pos_data))

    # 将两种样本合并
    datas, labels = [], []
    datas.extend(neg_data)
    datas.extend(pos_data)
    labels.extend(neg_label)
    labels.extend(pos_label)
    print("总样本个数:", len(datas))
    print("总标签个数:", len(labels))

    # 简单清洗一下样本
    temp = []
    for d in datas:
        d = clean_data(d)
        temp.append(d)

    datas = temp
    del temp

    # 至此把数据整理好了
    print("第一个样本:", datas[0])
    print("第一个样本的标签:", labels[0])

    # 如果想吧标签转为one_hot 装个keras　一行代码搞定
    labels = to_categorical(labels)
    print("第一个样本:", datas[0])
    print("第一个样本的标签:", labels[0])


class Model:
    def __init__(self, size_layer, num_layers, embedded_size,
                 dict_size, dimension_output, learning_rate):

        def cells(reuse=False):
            return tf.nn.rnn_cell.BasicRNNCell(size_layer, reuse=reuse)

        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.float32, [None, None])
        encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, embedded_size], -1, 1))












