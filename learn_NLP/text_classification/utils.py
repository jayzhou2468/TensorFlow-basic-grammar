# -*- coding: utf-8 -*-
# @Time    : 2019/7/17 下午6:33
# @Author  : Ryan
# @File    : utils.py

import sklearn.datasets
import numpy as np
import re
import collections
import random
from sklearn import metrics
from nltk.corpus import stopwords

english_stopwards = stopwords.words('english')

def clearString(string):
    string = re.sub('[^A-Za-z0-9]+', '', string)    # 除正则表达式外的字符去除
    string = string.split(' ')
    string = filter(None, string)   # filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回一个迭代器对象
    string = [y.strip() for y in string if y.strip() not in english_stopwards]   # 移除字符串头尾指定的字符（默认为空格）或字符序列
    string = ' '.join(string)
    return string.lower()

def separate_dataset(trainset, ratio=0.5):
    datastring = []
    datatarget = []
    for i in range(len(trainset.data)):
        data_ = trainset.data[i].split('/n')
        data_ = list(filter(None, data_))
        data_ = random.sample(data_, int(len(data_) * ratio))
        for n in range(len(data_)):
            data_[n] = clearString(data_[n])
        datastring += data_
        for n in range(len(data_)):
            datatarget.append(trainset.target[i])
    return datastring, datatarget

if __name__ == '__main__':
    trainset = sklearn.datasets.load_files(container_path='data', encoding='UTF-8')

    string = '\n'.join(trainset.data)
    temp = string.split('\n')
    print('*'*100)

    print(len(temp))


    separate_dataset(trainset, ratio=0.5)



















