#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2021/11/25 5:42 下午
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
from transformers import AutoTokenizer


class Config(object):
    def __init__(self, bert_name='bert-base-chinese', learning_rate=1e-5, batch_size=16, num_epochs=1, maxlen=16,
                 mode='last'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("device is ：%s" % self.device)

        self.bert_name = bert_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_name)

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.maxlen = maxlen

        self.mode = mode
        self.model_dir = "model_dir/" + "bert_" + mode + ".pt"

        self.class_num = 6
        self.label_dict = {
            "游戏": 0,
            "美食": 1,
            "科技": 2,
            "体育": 3,
            "旅游": 4,
            "汽车": 5
        }