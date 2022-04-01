#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2021/11/24 7:36 下午
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

class ACAutomatonModel(object):
    def __init__(self,filename):
        self._ac = TrieTree()
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                data_list = line.strip().split('\t')
                self._ac.addWord(data_list[0])
                self._ac.addWord(data_list[1])
        self._ac.make()

    def search(self, data):
        search_result = self._ac.search(data)
        return search_result

# Trie树的叶子结点
class Node(object):
    def __init__(self):
        self.next = {}
        self.fail = None  # 失败指针
        self.isWord = False # 匹配标识
# AC自动机
class TrieTree(object):
    '''
    Trie树+KMP算法
    '''
    def __init__(self):
        self.__root = Node() # trie树，也可看作根节点

    def addWord(self, word):
        tmp = self.__root
        for i in word:
            # 如果tmp.next的key中不存在i，则加入i，且i的值为Node()并将Node赋值给tmp
            tmp = tmp.next.setdefault(i, Node())
        tmp.isWord = True  # 模式串结束位的标志

    def make(self):
        '''
        构建失败路径，失败路径也是指向Trie树(节点)
        '''
        tmpQueue = []
        tmpQueue.append(self.__root)  # 不断的添加，再弹出Trie树
        while (len(tmpQueue) > 0):
            temp = tmpQueue.pop()  # temp 为Trie树，弹出Trie树
            p = None
            for k, v in temp.next.items():  # k为字符，v为子树(节点)，按广度获取当前字符及其指向的节点(子树)
                if temp == self.__root:  # 特殊处理：第二层要特殊处理，将这层中的节点的失败路径直接指向父节点(也就是root)。
                    temp.next[k].fail = self.__root
                else:
                    p = temp.fail  # k节点的父节点的失败路径指向的节点p，此时p不为None
                    while p is not None:  # k节点的父节点的失败路径不为空
                        if k in p.next:
                            temp.next[k].fail = p.next[k]  # 将当前路径失败指针指向父节点的失败指针中含有k的节点, 并跳出While循环
                            break
                        # k不在p.next，说明p中不含有子节点k，不能继续匹配，取p节点的failNode节点(继续向上回溯)继续进行匹配
                        p = p.fail  # 回溯，取p节点的fail节点
                    if p is None:  # 当p是None时 , 表示没有找到k的失败指针，则将k的失败指针指向root
                        temp.next[k].fail = self.__root
                tmpQueue.append(temp.next[k])  # 添加子树

    def search(self, content):
        '''
        返回列表，每个元素为匹配的模式串在句中的起止位置
        '''
        result = ""
        currentPosition = 0
        while currentPosition < len(content):
            word = content[currentPosition]
            endWordIndex = currentPosition
            p = self.__root
            while (word in p.next):  # 存在状态转移，开始匹配
                fail_flag = False
                if (p == self.__root) or fail_flag:
                    # 若当前节点是根且存在转移状态，则说明是匹配词的开头，记录词的起始位置
                    startWordIndex = currentPosition
                # 转移状态机的状态
                p = p.next[word]
                if p.isWord:  # 转移完成后若词语标志位为真，则记录起止位置
                    result = result + "#" +content[int(startWordIndex):int(endWordIndex)+1]
                    currentPosition = endWordIndex
                if p.next and endWordIndex+1 < len(content):  # 若下一状态还存在(即模式串还未到末尾)且句子没有结束，将结束位置右移一位
                    endWordIndex += 1
                    word = content[endWordIndex]  # 准备匹配新字符
                else: #
                    break
                while (word not in p.next) and (p != self.__root):  # 不存在状态转移且不为匹配词的开头，走失败路径
                    p = p.fail  # p走失败路径，转移到失败状态后返回第一个while继续查看word是否存在状态转移
                    fail_flag = True
                if fail_flag:
                    currentPosition = endWordIndex
            currentPosition = currentPosition + 1
        "#".join(list(set(result.split('#'))))  #去重操作，也可以
        return result.strip("#")