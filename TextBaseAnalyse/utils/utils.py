#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2021/11/24 11:31 上午
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
# 生成ngram
def generate_ngram(input_list, n):
    '''
    输入的是一个文本的字符串及n，输出的是n及n一下的所有ngram的结果
    比如input_list=abcd, n=3，返回的就是a,b,c,d,ab,bc,cd,abc,bcd的一个list。
    '''
    result = []
    for i in range(1, n+1):
        result.extend(zip(*[input_list[j:] for j in range(i)]))
    return result

# 得到停用词
def get_stopwords():
    with open('data/stopword.txt', 'r') as f:
        stopword = [line.strip() for line in f]
    return set(stopword)