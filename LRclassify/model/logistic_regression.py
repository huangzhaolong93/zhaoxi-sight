#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022/3/30 上午11:03
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
import numpy as np
import math
import random
from utils.utils import cal_accuracy

class LR_classify(object):

    def sigmoid(self, num_list):
        if isinstance(num_list, (int, float)):
            return 1/(1+math.exp(-num_list))
        else:
            result_list = []
            for num in num_list:
                if isinstance(num, (int, float)):
                    result_list.append(1/(1+math.exp(-num)))
                else:
                    print ("error type!")
                    return -1
            return result_list

    def LR_train(self, X_train, Y_train, train_step = 1000):
        #训练步数
        train_step = 10
        #初始化权重
        weights=[random.random() for _ in range(X_train.shape[1]+1)]
        #训练集扩展为广义矩阵
        X_train = np.hstack((X_train, np.ones(X_train.shape[0]).reshape(X_train.shape[0],1)))
        #定义梯度下降的step
        learning_rate = 0.001
        #对于每个样本
        sample_num = X_train.shape[0]

        for step in range(train_step):
            #预测
            pred = self.sigmoid(np.dot(X_train,weights))
            for i in range(sample_num):
                error = Y_train[i]- pred[i]
                weights = weights + learning_rate * error * X_train[i]
            if step % 1 == 0:
                acc = cal_accuracy([round(z) for z in pred], Y_train)
                print("训练过程第",step,"步的准确率为：",acc)

        self.weights = weights

    def LR_test(self, X_test):
        X_test = np.hstack((X_test, np.ones(X_test.shape[0]).reshape(X_test.shape[0],1)))
        weights = self.weights
        pred = self.sigmoid(np.dot(X_test,weights))

        return pred