#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022/3/30 上午11:26
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

from reader.read import spyder
from model.ngram import Ngram

if __name__ == "__main__":
    #定义一个爬虫，去抓取相关的文本
    spy = spyder()
    txt = spy.crawl('Sherlock Holmes')
    #对文本进行ngram
    ngram = Ngram()
    ngram.generate_ngram(txt)
    #自动生成gram
    n = 3
    start_word = "then"
    sentence_len = 20
    ngram.generate_ngram_sentence(n,start_word,sentence_len)