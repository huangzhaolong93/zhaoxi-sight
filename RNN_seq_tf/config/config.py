#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022/4/2 上午11:42
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


class Config(object):
    def __init__(self, data_type, model_type, test_row=365, batch_size=64, WINDOW_SIZE=60, shuffle_seed=7, epochs=500,
                 learning_rate=0.001):
        # 默认取10年的数据，用前9年的数据进行训练，最后1年的数据进行预测
        self.test_row = test_row
        self.batch_size = batch_size
        # 用某日前60天窗口数据作为输入预测该日数据
        self.WINDOW_SIZE = WINDOW_SIZE
        self.shuffle_seed = shuffle_seed
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.data_type = data_type
        self.model_type = model_type