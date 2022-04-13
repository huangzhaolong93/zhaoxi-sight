#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2021/11/24 11:26 上午
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
import math
#建立字典树的节点

#如果语料较大，构建一次trie非常的耗时，可以考虑进行存储，后面进行增量构建。
import pickle

def save_model(model, filename):
    with open(filename, 'wb') as fw:
        pickle.dump(model, fw)


def load_model(filename):
    with open(filename, 'rb') as fr:
        model = pickle.load(fr)
    return model


class Node(object):
    """
    建立字典树的节点
    """
    def __init__(self, char):
        self.char = char
        # 记录是否完成
        self.word_finish = False
        # 用来计数
        self.count = 0
        # 用来存放节点
        self.child = []
        # 方便计算 左右熵
        # 判断是否是后缀（标识后缀用的，也就是记录 b->c->a 变换后的标记）
        self.isback = False


class TrieNode(object):
    """
    建立前缀树，并且包含统计词频，计算左右熵，计算互信息的方法
    如果报错dict_list = [result[0][0]]，说明PMI_limit设置的太高了，导致建树失败
    """
    def __init__(self, node, PMI_limit=20, data=None):
        """
        初始函数，data为外部词频数据集
        """
        self.root = Node(node)
        self.PMI_limit = PMI_limit
        if not data:
            return
        node = self.root
        for key, values in data.items():
            new_node = Node(key)
            new_node.count = int(values)
            new_node.word_finish = True
            node.child.append(new_node)
            print(key, values)

    def add(self, word):
        """
        添加节点，但是由于正常的树是单向的，左邻熵不好计算，所以采用了一个trick，用a->b<-c 来表示 cba，具体实现是利用 self.isback 来进行判断
        """
        node = self.root

        for count, char in enumerate(word):
            found_in_child = False

            for child in node.child:
                if char == child.char:
                    node = child
                    found_in_child = True
                    break


            if not found_in_child:
                new_node = Node(char)
                node.child.append(new_node)
                node = new_node


            if count == len(word) - 1:
                node.count += 1
                node.word_finish = True

        length = len(word)
        node = self.root
        if length == 3:
            word = list(word)
            word[0], word[1], word[2] = word[1], word[2], word[0]

            for count, char in enumerate(word):
                found_in_child = False

                if count != length - 1:
                    for child in node.child:
                        if char == child.char:
                            node = child
                            found_in_child = True
                            break
                else:
                    for child in node.child:
                        if char == child.char and child.isback:
                            node = child
                            found_in_child = True
                            break

                if not found_in_child:
                    new_node = Node(char)
                    node.child.append(new_node)
                    node = new_node

                if count == len(word) - 1:
                    node.count += 1
                    node.isback = True
                    node.word_finish = True

    def search_one(self):
        """
        计算互信息: 寻找一阶共现，并返回词概率
        """
        result = {}
        node = self.root
        if not node.child:
            return False, 0

        total = 0
        for child in node.child:
            if child.word_finish is True:
                total += child.count

        for child in node.child:
            if child.word_finish is True:
                result[child.char] = child.count / total

        # 返回 p(a) 和 N
        return result, total

    def search_bi(self):
        """
        计算互信息: 寻找二阶共现，并返回 log2( P(a,b) / (P(a) * P(a)) 和 p(a,b))
        """
        result = {}
        node = self.root
        if not node.child:
            return False, 0

        total = 0
        one_dict, total_one = self.search_one()
        for child in node.child:
            for ch in child.child:
                if ch.word_finish is True:
                    total += ch.count

        for child in node.child:
            for ch in child.child:
                if ch.word_finish is True:
                    # PMI， log( p(a,b)/p(a)p(b) )
                    PMI = math.log(max(ch.count, 1), 2) - math.log(total, 2) - math.log(one_dict[child.char],
                                                                                        2) - math.log(one_dict[ch.char],
                                                                                                      2)
                    # 这里做了PMI阈值约束
                    if PMI > self.PMI_limit:

                        result[child.char + '_' + ch.char] = (PMI, ch.count / total)

        #返回 PMI 和 p(a,b)
        return result

    def search_left(self):
        """
        寻找左频次
        统计左熵， 并返回左熵 (bc - a 这个算的是 abc|bc 所以是左熵)
        """
        result = {}
        node = self.root
        if not node.child:
            return False, 0

        for child in node.child:
            for cha in child.child:
                total = 0
                p = 0.0
                for ch in cha.child:
                    if ch.word_finish is True and ch.isback:
                        total += ch.count
                for ch in cha.child:
                    if ch.word_finish is True and ch.isback:
                        p += (ch.count / total) * math.log(ch.count / total, 2)
                # 计算的是信息熵
                result[child.char + cha.char] = -p
        return result

    def search_right(self):
        """
        寻找右频次
        统计右熵，并返回右熵 (ab - c 这个算的是 abc|ab 所以是右熵)
        """
        result = {}
        node = self.root
        if not node.child:
            return False, 0

        for child in node.child:
            for cha in child.child:
                total = 0
                p = 0.0
                for ch in cha.child:
                    if ch.word_finish is True and not ch.isback:
                        total += ch.count
                for ch in cha.child:
                    if ch.word_finish is True and not ch.isback:
                        p += (ch.count / total) * math.log(ch.count / total, 2)
                # 计算的是信息熵
                result[child.char + cha.char] = -p
        return result

    def find_word(self, N):
        '''
        基于trie树的结构
        search_bi计算每个gram的PMI和p(a,b)，search_left计算每个gram的E(L)，search_right计算每个gram的E(R) ，注意此时E没有包含p(a,b)
        计算score = Agg(a,b) + Rni(a,b)，由于都有p(a,b)，所以计算Agg的时候，单独将p(a,b)传出来放在values[1]里，然后E(L)和E(R)的计算就先忽略了p(a,b)
        返回的结构是dict，key是"a_b"，value就是相应的值。
        '''
        bi = self.search_bi()
        left = self.search_left()
        right = self.search_right()

        result = {}
        for key, values in bi.items():
            d = "".join(key.split('_'))
            # 计算score
            result[key] = (values[0] + min(left[d], right[d])) * values[1]

        # 按score排序
        result = sorted(result.items(), key=lambda x: x[1], reverse=True)

        dict_list = [result[0][0]]
        add_word = {}
        new_word = "".join(dict_list[0].split('_'))

        add_word[new_word] = round(result[0][1],5)

        # 取前N个
        for d in result[1: N]:
            flag = True
            for tmp in dict_list:
                pre = tmp.split('_')[0]
                # 按score排序，如果存在词与词之间有交叠的部分，做如下处理：
                # 1、如果存在B_C,A_B,则将A_B删掉
                # 2、如果存在A_B,B_C,则B_C保留
                # 3、如果存在候选词子是当前词的子集，说明当前词组合的太大，删去。
                if d[0].split('_')[-1] == pre or "".join(tmp.split('_')) in "".join(d[0].split('_')):
                    flag = False
                    break
            if flag:
                new_word = "".join(d[0].split('_'))
                add_word[new_word] = round(d[1],5)
                dict_list.append(d[0])

        return result, add_word