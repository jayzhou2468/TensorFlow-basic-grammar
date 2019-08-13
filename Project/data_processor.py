# -*- coding: utf-8 -*-
# @Time    : 2019/8/2 下午2:32
# @Author  : Ryan
# @File    : data_processor.py

import tensorflow as tf
import numpy as np
import codecs
from load_path import load_path_func
import string
import re
import json
import os
import math


# 1. 读取数据集corpus
def data_process(path):
    # 判断文件扩展名是否属于给定拓展名
    def endWith(s, *endstring):
        array = map(s.endswith, endstring)
        if True in array:
            return True
        else:
            return False

    # 加载文件并判断
    def load_path_func(input_path):
        try:
            if os.path.exists(input_path):

                if (endWith(input_path, '.txt', '.csv', 'data')):
                    # # print(os.path.abspath(imput_path))
                    # loc = re.findall(r'\\', input_path)[-1]
                    # file_name = input_path([loc, -1])
                    print("—————————————")
                    print("文件读入成功!")
                    print("—————————————")
                    return os.path.abspath(input_path)
                else:
                    print("不识别文件后缀!")
            else:
                print("该文件不存在!")
        except:
            print("读入文件发生了异常!")

# 2. 读取data并分离问句和答句

    questions = []
    answers = []
    try:
        with codecs.open(path, 'r', 'utf-8') as file:
            for each in file.readlines():
                # 分割问答句
                each = each.replace('\n', '')     # 去掉换行符
                q_a_split = each.split('=')
                inputs_ques = q_a_split[0]
                inputs_answ = q_a_split[1]
                # 问、答句分别存入列表
                questions.append(inputs_ques)
                answers.append(inputs_answ)
            return questions, answers

            # data_len = len(questions)           # 数据集大小为116600
            # print(data_len)

    except UnicodeDecodeError as e:
        print('UnicodeDecodeError')

# 3. 清洗：去掉长度为0、标点符号（可选）
def clean_data(data, flag):

    # flag==1执行本函数
    if(flag == 1):
        # 去掉列表中空字符串
        filter(None, data)

        # 去标点符号，只保留中文、大小写字母和阿拉伯数字
        res = []
        reg = "[^0-9A-Za-z\u4e00-\u9fa5]"
        for text in data:
            new_text = re.sub(reg, '', text)
            res.append(new_text)
        return res
        # print(res)

        # 只能去英文标点符号
        # res = []
        # for text in data:
        #     new_text = ''.join(c for c in text if c not in string.punctuation)
        #     res.append(new_text)
        # print(res)
    else:
        pass


# 4. 读取字典

def read_vocab(path):
    load_path_func(path)
    with codecs.open(path, 'r', 'utf-8') as file:
        lines = file.readlines()
        vocab = [line.replace('\n', '') for line in lines]
    vocab2id = {}
    for i, v in enumerate(vocab):
        vocab2id[v] = i
    return vocab2id
    # print(vocab2id)


# 5. 转换str-index

def convert(data, vocab):
    index = []
    for each_sentence in data:
        temp = [vocab.get(i, 0) for i in each_sentence]
        index.append(temp)
    return index


# # 6. 将list转为json，保存数据为txt
# def save(filename, contents):
#
#     func = lambda list_inner : ' '.join(list_inner)
#     out_list = map(func, contents)
#
#     # jsonList = json.dumps(contents, ensure_ascii=False)
#     # out_str = "".join(contents)
#     # out_str = out_str.replace(',', ' ')
#     with codecs.open(filename, 'w', 'utf-8') as file:
#         for i in out_list:
#             file.write(str(i))
#         file.close()

def save_txt(filename, questions):
    # print(questions)
    total_str = ''
    for s in questions:
        temp = ''
        for c in s:
            temp += str(c)
            temp += ' '
        total_str += temp
        total_str += '\n'
    with codecs.open(filename, 'w', encoding='utf8') as f:
        f.write(total_str)
        f.close()


# 7. 取某一块batch
def get_batch(batch_size, data, num):
    batch_step = math.ceil(len(data) / batch_size)
    item_list = []
    for index in range(0, len(data), batch_step):
        temp = data[index:index+batch_step]
        item_list.append(temp)
    return item_list[num-1]


# 8. 按顺序对应shuffle两个列表
def shuffleList(ques, answ):
    zipList = [i for i in zip(ques, answ)]
    np.random.shuffle(zipList)
    ques[:], answ[:] = zip(*zipList)
    return ques, answ



if __name__ == '__main__':

    # 1.加载数据 # 2.分离问答句
    questions, answers = data_process('data/qinyun_chat.txt')
    # print(questions)

    # 3.数据清洗
    # print("是否去除数据集中标点符号?")
    # bool_clear = input("是y,否n:")
    # if (bool_clear == 'y' or 'yes' or '是'):
    questions = clean_data(questions, 1)
    answers = clean_data(answers, 1)
    # elif (bool_clear == 'n' or 'no' or '否'):
    #     pass
    # else:
    #     print("输入有误!")

    # 4.读取字典
    vocab = read_vocab('data/vocab.data')

    # 5.转换index-str
    questions = convert(questions, vocab)
    answers = convert(answers, vocab)

    # 6.保存数据
    # save('./data/questions.txt', questions)
    # save('./data/answers.txt', answers)
    print(questions)
    save_txt('./data/questions.txt', questions)
    save_txt('./data/answers.txt', answers)

    # 7.取某一块batch
    batch_size = 128
    item_list = get_batch(batch_size, questions, 5)
    print(item_list)

    # 8.shuffle
    shuffleList(questions, answers)






