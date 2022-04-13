#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022/4/2 上午11:27
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
import tushare
import tensorflow as tf
'''
基于tushare模块，下载相应的股票数据
'''

def load_data(config):
    global WINDOW_SIZE
    WINDOW_SIZE = config.WINDOW_SIZE
    if config.data_type == "MaoTai":
        tushare_id = "600406"
    elif config.data_type == "YiLi":
        tushare_id = "600887"
    stock_data = tushare.get_k_data(tushare_id, ktype='D', start='2010-01-01', end='2020-01-01')
    train_row = stock_data.shape[1] - config.test_row
    #将Pandas DataFrame 转成 序列数据 的格式
    ds_train = generate_date(stock_data.iloc[0:train_row,3].values, config)

    ds_test_data = tf.data.Dataset.from_tensor_slices(tf.constant(stock_data.iloc[train_row:,3].values,dtype = tf.float32)) \
       .window(WINDOW_SIZE,shift=1).flat_map(batch_dataset)

    ds_test = ds_test_data.batch(config.batch_size)

    test_label = stock_data.iloc[train_row + WINDOW_SIZE:, 3].values
    return ds_train, ds_test, test_label

def batch_dataset(dataset):
    dataset_batched = dataset.batch(WINDOW_SIZE, drop_remainder=True)
    return dataset_batched

def generate_date(input_data, config):
    ds_data = tf.data.Dataset.from_tensor_slices(tf.constant(input_data, dtype=tf.float32)) \
        .window(WINDOW_SIZE, shift=1).flat_map(batch_dataset)
    ds_label = tf.data.Dataset.from_tensor_slices(tf.constant(input_data[WINDOW_SIZE:], dtype=tf.float32))
    return tf.data.Dataset.zip((ds_data, ds_label)).batch(config.batch_size).shuffle(config.shuffle_seed)