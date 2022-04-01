#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022/3/30 上午10:58
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
from sklearn.datasets import load_iris
import random

def load_data():

    iris = load_iris()
    X = iris.data
    Y = iris.target
    #特征离散化,多分类转二分类
    X = X[Y<=1].round()
    Y = Y[Y<=1]
    data_num = X.shape[0]
    index_list = [random.random() >= 0.2 for _ in range(data_num)]
    train_index_list = []
    test_index_list = []
    for i in range(len(index_list)):
        if index_list[i]:
            train_index_list.append(i)
        else:
            test_index_list.append(i)

    X_train = X[train_index_list]
    X_test = X[test_index_list]
    Y_train = Y[train_index_list]
    Y_test = Y[test_index_list]

    return X_train, X_test, Y_train, Y_test