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
///----------------------=============
"""
from word_discovery import word_discovery
from ac_automaton import ac_automaton
from bert_classify import bert_classify
import warnings

warnings.filterwarnings("ignore")
"""
本函数会对github上一些公开的数据集做一些基础的文本分析。
1、新词发现：对某一个领域的文章标题进行该领域的新词发现
2、ac自动机打标：对给定的标签集合，对文章标题进行自动打标
3、bert分类：对文章标题进行简单的领域分类
"""
if __name__ == "__main__":
    data_path = "data/ugc_title.txt"

    # -----------新词发现-----------
    '''
    print('\n ----- 新词发现 -----\n')
    model_dir = "model_dir/trie.pkl"
    theme = "游戏"  # "游戏","美食","科技","体育","旅游","汽车"
    topN = 10
    word_discovery(theme,data_path, model_dir, topN)
    '''

    # -----------AC自动机进行多模匹配-----------
    '''
    print('\n ----- AC自动机 -----\n')
    tag_path = "data/game_knowledge.txt"
    result_path = "data/game_tag_result.txt"
    ac_automaton(data_path, tag_path, result_path)
    '''
    # -----------bert进行文本分类-----------
    print('\n ----- 文本分类 -----\n')
    # mode = "last"，使用最后1层的cls
    # mode = "last_avg"，使用最后1层所有向量的均值
    # mode = "last_4_cls_avg"，使用最后4层的cls的均值
    # mode = "last_4_avg"，使用最后4层的所有向量的均值的均值
    result = ""
    for mode in ['last', 'last_avg', 'last_4_cls_avg', 'last_4_avg']:
        result_str = bert_classify(data_path, mode)
        result = result + result_str
    print('\n ----- 最终结果 -----\n')
    print(result)