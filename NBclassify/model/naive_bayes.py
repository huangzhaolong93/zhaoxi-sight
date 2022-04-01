#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022/3/30 下午2:36
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

class NB_classify(object):
    def NB_train(self, X_train, Y_train):
        '''
        在已知X和Y的情况下，构建联合分布P(X,Y),即参数计算，计算方式为统计
        根据贝叶斯公式，计算P(X,Y)=P(X|Y)*P(Y)
        '''
        p_xy = []
        #对于每个类别
        p_y = {}
        for k in set(Y_train):
            total_xi_y = list(Y_train).count(k)
            print("对于类别为",k,"的样本共有",total_xi_y,"个")
            #计算P(Y)
            p_y.setdefault(k,list(Y_train).count(k)/Y_train.shape[0])
            p_x_y_sub = []
            #对每个维度的特征，计算P(x|Y)
            for i in range(X_train.shape[1]):
                #对于每一维特征,统计当前类别下样本个数,得到P(x|Y)
                p_x_sub_y_sub = {}
                print("对于类别为",k,"的样本，第",i,"维的离散特征共有",len(set([(round(item)) for item in X_train[[l for l in range(len(Y_train)) if Y_train[l] == k],i]])),"种")
                #对每个维度下特征的取值，计算p(x=xi|Y)
                for j in set(X_train[[l for l in range(len(Y_train)) if Y_train[l] == k],i]):
                    #统计每个类别下，每个特征占该类别样本个数的比例,并做贝叶斯平滑
                    print("对于类别为",k,"的样本，第",i,"维特征为",j,"的样本共有",list(X_train[[l for l in range(len(Y_train)) if Y_train[l] == k],i]).count(j), "个")
                    #计算p(y)*p(x|y)
                    p_x_sub_y_sub.setdefault(j,p_y[k]*(list(X_train[[l for l in range(len(Y_train)) if Y_train[l] == k],i]).count(j)+1)/(total_xi_y+len(set(X_train[[l for l in range(len(Y_train)) if Y_train[l] == k],i]))))
                p_x_y_sub.append(p_x_sub_y_sub)
            '''
            最后得到P(X|Y)的模型参数，即矩阵M(xy)，其中矩阵M为m行n列,m为类别个数，n为特征维数。每个元素是一个字典，字典的key是该维特征的取值，字典的value是概率
            例如矩阵M中的元素M_ij，M_ij是一个字典。
            M_ij的key是第j维特征的所有取值，例如k_j
            M_ij的value是第j维特征取值为k_j时，类别为i的概率P(xj=k_j|i)，也就是所有类别为i的样本下，第j维特征为k_j的概率v_ij。
            v_ij = 所有类别为i的样本下，第j维特征为k_j的样本数数/所有类别为i的样本数
            '''
            p_xy.append(p_x_y_sub)
        print("P(Y)的先验概率：\n",p_y)
        print("P(X,Y)的联合概率：\n",p_xy)
        self.p_xy = p_xy

    def NB_test(self, X_test):
        p_xy = self.p_xy
        pred = []
        #对于每个样本
        for z in range(len(X_test)):
            #对于每个类别
            pj = [1]*3
            for j in range(3):
                #对于每个特征,通过联合概率分布，计算每个特征下得到的类别P(yj|xi)
                for i in range(len(X_test[z])):
                    if X_test[z][i] in p_xy[j][i]:
                        p_xi_yj = p_xy[j][i][X_test[z][i]]
                    else:
                        #如果训练集没有出现过该特征，则说明构建的P(X,Y)联合分布并不准确，生成式模型需要训练样本足够大，足够全
                        p_xi_yj = 0.0000001
                    #计算全部特征到每个类别的概率P(yj|X)，这里是认为各特征对类别是独立的，可以连乘
                    pj[j] = pj[j]*p_xi_yj
            #根据概率最大计算最可能的类别，即 argmax P(yj|X)
            pred.append(pj.index(max(pj)))

        return pred