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
from reader.read import load_data
from utils.utils import cal_accuracy
from model.logistic_regression import LR_classify

if __name__ == "__main__":

    acc = 0

    for i in range(1):
        X_train, X_test, Y_train, Y_test = load_data()
        lr = LR_classify()
        lr.LR_train(X_train, Y_train)
        pred = lr.LR_test(X_test)
        acc += cal_accuracy([round(z) for z in pred], Y_test)

    acc = acc/1

    print("测试集准确率:",acc)