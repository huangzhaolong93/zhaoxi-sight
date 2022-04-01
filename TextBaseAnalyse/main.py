#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2021/11/24 10:20 上午
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
import torch.nn as nn
from word_discovery import word_discovery
from ac_automaton import ac_automaton
from config.config import Config
from reader.read import UgcData
from torch.utils.data import DataLoader
from transformers import AdamW
from model.bert_base import UgcClassification
from train import train

"""
本函数会对github上一些公开的数据集做一些基础的文本分析。
1、新词发现：对某一个领域的文章标题进行该领域的新词发现
2、ac自动机打标：对给定的标签集合，对文章标题进行自动打标
3、
"""
if __name__ == "__main__":
    data_path = "data/ugc_title.txt"
    # ---------------新词发现---------------
    '''
    theme = "游戏"  # "游戏","美食","科技","体育","旅游","汽车"
    topN = 10
    word_discovery(theme , topN)
    '''
    # ---------------AC自动机进行多模匹配---------------
    '''
    ac_automaton(data_path)
    '''

    # ---------------定义基础参数---------------
    bert_name = 'bert-base-chinese'  # 'hfl/chinese-bert-wwm-ext'
    learning_rate = 1e-5
    batch_size = 16
    num_epochs = 1
    max_len = 32

    config = Config(bert_name, learning_rate, batch_size, num_epochs, max_len)

    # ---------------处理数据---------------
    ugc_data = UgcData(data_path, config)
    train_length = int(ugc_data.__len__() * 0.8)
    valid_length = int(ugc_data.__len__() * 0.1)
    test_length = ugc_data.__len__() - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(ugc_data, (
    train_length, valid_length, test_length))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)

    # ---------------定义模型、损失以及优化器---------------
    model = UgcClassification(config).to(config.device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    loss_func = nn.CrossEntropyLoss()

    print('\n ----- Training -----\n')
    train(config, model, optimizer, train_dataloader, loss_func)
    print('\n ----- Eval -----\n')