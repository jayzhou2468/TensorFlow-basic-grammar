# -*- coding: utf-8 -*-
# @Time    : 2019/8/7 上午8:52
# @Author  : Ryan
# @File    : load_path.py

'''~/wzy/nmt/data/dsadasds.txt'
    '~/zjx/nmt/data/dasdasdsad.txt'
    1. 判断txt
    2. 判断路径
    2.1  pathlib
    3. data/ data/在当前目录, pass
    4. ~/upper_dir/... /data/..., project (father dir), actua_dir + Project-> absolute dir'''

from pathlib import Path
import os
import sys
import re

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

            if(endWith(input_path, '.txt', '.csv', 'data')):
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


