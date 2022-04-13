#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022/4/12 上午11:16
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

def eval(config, model, eval_dataloader):
    ##------------------eval-------------------
    model.eval()
    correct = torch.zeros(1).squeeze().to(config.device)
    total = torch.zeros(1).squeeze().to(config.device)
    with torch.no_grad():
        for i, data in enumerate(tqdm(eval_dataloader)):
            content, label = data
            content = list(content)
            # tokenize the data text
            inputs = config.tokenizer(content, max_length=config.maxlen, padding='max_length', truncation=True, return_tensors='pt')

            # move data to device
            input_ids = inputs['input_ids'].to(config.device)
            token_type_ids = inputs['token_type_ids'].to(config.device)
            attention_mask = inputs['attention_mask'].to(config.device)
            labels = label.to(config.device)

            # forward and backward propagations
            logits = model(input_ids, attention_mask, token_type_ids)
            predict = torch.argmax(logits, 1)

            correct += (predict == labels).sum().float()
            total += len(labels)

    return correct/total