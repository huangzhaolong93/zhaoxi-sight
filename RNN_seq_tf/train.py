#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022/4/2 下午2:39
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

import datetime
from pathlib import Path

import tensorflow as tf
from utils.loss import MSPE

def train_model(ds_train, model, config):
    # 在 Python3 下建议使用 pathlib 修正各操作系统的路径
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = str(Path('./data/autograph/' + stamp))

    tb_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    #如果loss在100个epoch后没有提升，学习率减半。
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor = 0.5, patience = 100)
    #当loss在200个epoch后没有提升，则提前终止训练。
    stop_callback = tf.keras.callbacks.EarlyStopping(monitor = "loss", patience= 200)
    callbacks_list = [tb_callback,lr_callback,stop_callback]

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    loss = MSPE(config, name = "MSPE")
    model.compile(optimizer=optimizer, loss=loss, metrics=["mse", "mae"])

    history = model.fit(ds_train, epochs=config.epochs, callbacks = callbacks_list)
    return model