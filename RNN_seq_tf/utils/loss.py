#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022/4/2 下午2:29
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
import tensorflow as tf

#自定义损失函数，MSE改进下，loss乘上sqrt(batch_size)

class MSPE(tf.keras.losses.Loss):
    def __init__(self, config, name = "MSPE"):
        self.batch_size = config.batch_size
        super(MSPE, self).__init__()

    def call(self, y_true, y_pred):
        err_percent = (y_true - y_pred)**2 * self.batch_size**0.5
        mean_err_percent = tf.reduce_mean(err_percent)
        return mean_err_percent