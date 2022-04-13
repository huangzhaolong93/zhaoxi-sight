#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022/4/1 上午10:40
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
from config.config import Config
from reader.read import UgcData
from torch.utils.data import DataLoader
from transformers import AdamW
from model.bert_base import UgcClassification
from utils.bert_train import train
from utils.bert_eval import eval


def bert_classify(data_path, mode):
    # ---------------定义基础参数---------------
    bert_name = 'bert-base-chinese'  # 'hfl/chinese-bert-wwm-ext'
    learning_rate = 1e-5
    batch_size = 128
    num_epochs = 6
    max_len = 32
    assert mode in ['last', 'last_avg', 'last_4_cls_avg', 'last_4_avg']
    config = Config(bert_name, learning_rate, batch_size, num_epochs, max_len, mode)

    # ---------------处理数据---------------
    ugc_data = UgcData(data_path, config)
    train_length = int(ugc_data.__len__() * 0.8)
    eval_length = ugc_data.__len__() - train_length
    print('训练集长度：%d，验证集长度：%d' % (train_length, eval_length))
    train_dataset, eval_dataset = torch.utils.data.random_split(ugc_data, (train_length, eval_length))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)

    # ---------------定义模型、损失以及优化器---------------
    model = UgcClassification(config).to(config.device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    loss_func = nn.CrossEntropyLoss()

    print('\n ----- Training -----\n')
    train(config, model, optimizer, train_dataloader, loss_func)
    print('\n ----- Eval -----\n')
    acc = eval(config, model, eval_dataloader)
    print("Use mode【%s】 Accuracy is : %.3f" % (mode, acc))

    result_str = 'Use mode【' + mode + '】 Accuracy is : ' + str(acc.cpu().detach().data.numpy()) + "\n"
    return result_str