#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2021/11/24 2:44 下午
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
import os
from reader.read import load_data
from model.WordDiscovery import TrieNode, load_model, save_model
from utils.utils import generate_ngram
from tqdm import tqdm


def word_discovery(theme, data_path, model_dir=None, topN=10):
    theme_list = ["游戏", "美食", "科技", "体育", "旅游", "汽车"]
    if theme not in theme_list:
        print("error， please choose right theme")
        return

    if os.path.exists(model_dir):
        print("Load Old Tree")
        root = load_model(model_dir)
    else:
        print("Build New Tree")
        root = TrieNode('*', PMI_limit=10)
    data_list = load_data(data_path, use_jieba=True)
    '''
    将句子插入到trie中，分两步，1、转成ngram，2、插入到trie树中。
    由于只计算2个token能否成词，所以n=3，因为需要包含一个邻字。
    由于输入最多只有3gram，所以不算根节点，trie树只有3层。
    '''
    print('------> 插入节点')
    step = 0
    for i, value in enumerate(tqdm(data_list)):
        word_list, label = value
        # 建树有性能问题，可以用step限制
        if label == theme and step <= 10000:
            ngrams = generate_ngram(word_list, 3)
            for d in ngrams:
                root.add(d)
            step = step + 1
    print(step)
    print('------> 插入成功')

    save_model(root, model_dir)

    result, add_word = root.find_word(topN)
    print(add_word)
    print("主题【%s】的新词发现结果为：%s" % (theme, ','.join(list(add_word.keys()))))