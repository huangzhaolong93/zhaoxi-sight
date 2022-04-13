#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022/4/2 上午11:26
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
from config.config import Config
from reader.read import load_data
from train import train_model
from model.rnn import build_rnn_model
from model.lstm import build_lstm_model
from utils.draw import draw

if __name__ == "__main__":
    print("tensorflow version:", tf.__version__)

    # 数据来源
    data_type = "YiLi"  # MaoTai, YiLi
    print("data_type: %s" % (data_type))
    # 模型选择
    model_type = "lstm"  # rnn, lstm
    print("model_type: %s" % (model_type))

    # 取10年的数据，用前9年的数据进行训练，最后1年的数据进行预测
    test_row = 365
    batch_size = 64
    shuffle_seed = 7
    # 用某日前60天窗口数据作为输入预测该日数据
    WINDOW_SIZE = 60
    epochs = 500
    learning_rate = 1e-3

    config = Config(data_type, model_type, test_row, batch_size, WINDOW_SIZE, shuffle_seed, epochs, learning_rate)

    ds_train, ds_test, real_stock_price = load_data(config)

    print('\n ----- model define -----\n')
    if config.model_type == "rnn":
        model = build_rnn_model()
    elif config.model_type == "lstm":
        model = build_lstm_model()
    else:
        model = build_rnn_model()

    print('\n ----- train -----\n')
    model = train_model(ds_train, model, config)

    '''
    用最后一年的数据ds_test进行预测，同样是用60天的数据预测第61天到数据，并和真实的第61天的数据进行比较。
    '''
    print('\n ----- pred -----\n')
    predicted_stock_price = model.predict(ds_test)

    draw(predicted_stock_price, real_stock_price, config)