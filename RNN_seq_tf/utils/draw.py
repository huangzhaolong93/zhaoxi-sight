#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022/4/2 下午4:23
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
import matplotlib.pyplot as plt


def draw(predicted_stock_price, real_stock_price, config):
    label_str = config.data_type + " Stock Price"
    predict_label_str = "Predicted " + config.data_type + " Stock Price"
    title_str = config.data_type + " Stock Price Prediction"
    y_label_str = config.data_type + " Stock Price"
    picture_str = config.data_type + "_" + config.model_type + ".png"

    plt.plot(real_stock_price, color='red', label=label_str)
    plt.plot(predicted_stock_price, color='blue', label=predict_label_str)
    plt.title(title_str)
    plt.xlabel('Time')
    plt.ylabel(y_label_str)
    plt.legend()
    # plt.show()
    plt.savefig(picture_str)