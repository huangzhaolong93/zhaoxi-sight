#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2021/11/25 7:47 下午
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
from transformers import BertTokenizer, AdamW, BertModel
# 模型准备
class UgcClassification(nn.Module):
    def __init__(self, config):
        super(UgcClassification, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_name)
        self.linear = nn.Linear(in_features=768, out_features=config.class_num)
        self.mode = config.mode

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
        assert self.mode in ['last', 'last_avg', 'last_4_cls_avg', 'last_4_avg']

        if self.mode == "last":
            output = outputs[0][:,0,:] #last_hidden_states的第一个元素
        elif self.mode == "last_avg":
            output = torch.sum(outputs[0]*attention_mask.float().unsqueeze(2),1)
            output_mask = torch.sum(attention_mask.float().unsqueeze(2),1)
            output = output / output_mask #取[CLS]和所有word embedding和[SEP]求AVG，进行分类
        elif self.mode == "last_4_cls_avg":
            output = torch.cat([x[:,0].unsqueeze(1) for x in outputs[2][-4:]],dim=1)
            output = torch.mean(output, dim=1)
        elif self.mode == "last_4_avg":
            output_list = []
            for x in outputs[2][-4:]:
                output = torch.sum(x*attention_mask.float().unsqueeze(2),1)
                output_mask = torch.sum(attention_mask.float().unsqueeze(2),1)
                output = output / output_mask
                output_list.append(output.unsqueeze(1))
            output = torch.cat(output_list,dim=1)
            output = torch.mean(output, dim=1)
        else:
            output = outputs[0][:,0,:]

        logits = self.linear(output)

        return logits

