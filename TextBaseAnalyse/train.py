#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2021/11/25 7:57 下午
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
from tqdm import tqdm

def train(config, model, optimizer, train_dataloader, loss_func):
    for epoch in range(config.num_epochs):
        ##------------------train-------------------
        model.train()
        running_loss = 0.0
        for i, data in enumerate(tqdm(train_dataloader)):
            content, label = data
            content = list(content)
            # tokenize the data text
            inputs = config.tokenizer(content, max_length=config.maxlen, padding='max_length', truncation=True, return_tensors='pt')

            # move data to device
            input_ids = inputs['input_ids'].to(config.device)
            token_type_ids = inputs['token_type_ids'].to(config.device)
            attention_mask = inputs['attention_mask'].to(config.device)
            label = label.to(config.device)

            # forward and backward propagations
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = loss_func(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        print('epoch%d, loss: %.3f' % (epoch + 1, running_loss / len(train_dataloader)))