#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2021/11/24 8:11 下午
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
from tqdm import tqdm
from model.ACAutomaton import ACAutomatonModel

def ac_automaton(data_path, tag_path, result_path):

    acam = ACAutomatonModel(tag_path)
    print("-----------build ac automaton------------")
    data = []
    with open(data_path, 'r') as f:
        for i, line in enumerate(tqdm(f)):
            data_list = line.strip().split('\t')
            sentence = data_list[0]
            tag = acam.search(sentence)
            data.append((sentence, tag))
    f.close()

    print("-----------write result------------")
    with open(result_path, 'w+') as f:
        for sentence, tag in data:
            str = sentence+'\t'+tag+'\n'
            f.write(str)
    f.close()