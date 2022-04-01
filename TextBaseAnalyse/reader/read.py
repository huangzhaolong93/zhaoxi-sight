#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2021/11/24 10:21 上午
@author: Zhaolong Huang
@email: zhaolong.hzl@alibaba-inc.com
///
///                   _ooOoo_
///                  o8888888o
///                  88" . "88
///                  (| -_- |)
///                  O\  =  /O
///               ____/`---'\____
///             .'  \\|     |//  `.
///            /  \\|||  :  |||//  \
///           /  _||||| -:- |||||-  \
///           |   | \\\  -  /// |   |
///           | \_|  ''\---/''  |   |
///           \  .-\__  `-`  ___/-. /
///         ___`. .'  /--.--\  `. . __
///      ."" '<  `.___\_<|>_/___.'  >'"".
///     | | :  `- \`.;`\ _ /`;.`/ - ` : | |
///     \  \ `-.   \_ __\ /__ _/   .-` /  /
///======`-.____`-.___\_____/___.-`____.-'======
///                   `=---='
///^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
///
///         佛祖保佑       永无BUG
///
///  本模块已经经过开光处理，绝无可能再产生bug
///
///=============================================
"""
import torch
import jieba
from torch.utils.data import Dataset

def load_data(filename, use_jieba=False):
    """
    加载语料，以句子维度，用jieba分词断句，用于生成ngram。不用分词也可以直接使用char，只不过最后的结果就只有2个字了。
    """
    data = []
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            data_list = line.strip().split('\t')
            if use_jieba:
                text = list(jieba.cut(data_list[0].strip(), cut_all=False))
            else:
                text = data_list[0]
            label = data_list[1]
            data.append((text, label))
    return data

class UgcData(Dataset):
    def __init__(self, filename, config):
        self.data = load_data(filename)
        self.label_dict = config.label_dict

    def __getitem__(self, idx):
        content = self.data[idx][0]
        label = torch.tensor(self.label_dict[self.data[idx][1]])
        return content, label

    def __len__(self):
        return len(self.data)