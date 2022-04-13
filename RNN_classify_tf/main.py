#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022/4/13 下午3:50
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
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers, models

batch_size = 128 # 批量大小
total_words = 10000 # 词汇表大小 N_vocab
max_seq_len = 80 # 句子最大长度 s，大于的句子部分将截断，小于的将填充
embedding_len = 64 # 词向量特征长度 f
shuffle_seed = 7

# 加载 IMDB 数据集，此处的数据采用数字编码，一个数字代表一个单词
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)

# 截断和填充句子，使得等长，此处长句子保留句子后面的部分，短句子在前面填充
x_train = keras.preprocessing.sequence.pad_sequences(x_train,maxlen=max_seq_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test,maxlen=max_seq_len)

# 构建数据集，打散，批量，并丢掉最后一个不够 batch_size 的 batch
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(shuffle_seed).batch(batch_size, drop_remainder=True)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(batch_size, drop_remainder=True)

'''
模型定义
'''
tf.keras.backend.clear_session()

model = models.Sequential()

x_input = layers.Input(shape = [max_seq_len],dtype = tf.float32)
x = layers.Embedding(total_words, embedding_len)(x_input)
x = layers.GRU(32,return_sequences = True)(x)
x = layers.GRU(16)(x)
x_output = layers.Dense(1)(x)
prob = tf.sigmoid(x_output)

model = models.Model(inputs = [x_input],outputs = [x_output])
model.summary()

model.compile(
    optimizer = optimizers.Adam(0.001),
    loss = losses.BinaryCrossentropy(),
    metrics=['accuracy']
    )

# epochs不用太高，容易过拟合，训练集也才25000条
model.fit(db_train, epochs=5)
# 测试
model.evaluate(db_test)