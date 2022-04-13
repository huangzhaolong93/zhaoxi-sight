#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022/4/2 上午11:32
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
'''
定义网络模型
'''
def build_rnn_model():
    tf.keras.backend.clear_session()

    x_input = tf.keras.layers.Input(shape = (None,1),dtype = tf.float32)
    x = tf.keras.layers.SimpleRNN(30,return_sequences = True)(x_input)
    x = tf.keras.layers.SimpleRNN(10)(x)
    x_output = tf.keras.layers.Dense(1)(x)

    model = tf.keras.models.Model(inputs = [x_input],outputs = [x_output])
    model.summary()

    return model

