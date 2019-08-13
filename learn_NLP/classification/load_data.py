# -*- coding: utf-8 -*-
# @Time    : 2019/7/23 下午4:17
# @Author  : Ryan
# @File    : load_data.py

import glob


def read_data(path):
    with open(path, 'r', encoding='utf8') as f:
        line = f.read()
        line += '\n'
    return line


if __name__ == "__main__":
    neg_path = glob.glob('./data/neg/*.txt')
    pos_path = glob.glob('./data/pos/*.txt')

    neg_data = ''
    for path in neg_path:
        temp1 = read_data(path)

        neg_data += temp1
    pos_data = ''
    for path in pos_path:
        temp2 = read_data(path)

        pos_data += temp2


    # reviews = pos_data.split('\n')
    # print(len(reviews))
    # labels = '1\n'*len(reviews)

    print(pos_data)
    with open('./data/train_reviews.txt', 'a', encoding='utf8') as f:
        f.write(pos_data)

    # with open('./data/train_labels.txt', 'a', encoding='utf8') as f:
    #     f.write(labels)





