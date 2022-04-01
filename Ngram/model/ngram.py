#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022/3/30 上午11:30
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
import re

class Ngram(object):
    def __init__(self):
        #初始化Ngram中N的取值
        gram_N = [1,2,3,4,5]
        self.gram_N=gram_N
        #gram_list的每个元素是一个dict，这个dict包含了该Ngram下的gram和次数
        gram_list = [[] for i in gram_N]
        self.gram_list = gram_list

    def get_txt(self,txt):
        '''
        将抓取的内容，拆分成各个word
        '''
        #对这个文本取单词,所有单词构成一个list，并且过滤掉所有空
        words = re.split('[^A-Za-z]+', txt.lower())
        words = list(filter(None, words))
        print ("文本中所有单词的个数：",len(words))
        return words

    def generate_ngram(self, txt):
        '''
        将抓取的内容，存成gram_list
        gram_list里的每个元素是一个对应一个n的dict
        dict的key是ngram的gram,value是次数
        '''
        words = self.get_txt(txt)

        gram_N = self.gram_N
        gram_list = self.gram_list
        gram_dict = [{} for i in gram_N]

        for n in gram_N:
            for i in range(len(words)-n+1):
                word_group = tuple(words[i:i+n])
                if word_group in gram_dict[n-1].keys():
                    gram_dict[n-1][word_group] += 1
                else:
                    gram_dict[n-1].update({word_group:1})
            '''
            将生成好的ngram的dict，根据gram出现的次数进行排序
            '''
            gram_list[n-1] = sorted(gram_dict[n-1].items(), key = lambda x:-x[1])
            print(n,"gram里排名前五的元素",gram_list[n-1][:5])

        self.gram_list = gram_list

    def generate_ngram_sentence(self, n=2, start_word = "you", length = 15):

        gram_list = self.gram_list

        print ("generating sentence...\n")
        '''
        根据起始词，生成ngram的前序序列
        例如如果是4gram，，起始词是"you"
        先根据2gram+'you'，找到'are'
        在根据3gram+'you','are'找到'not'
        最后得到前序输入'you','are','not'开始4gram生成下一个词
        '''
        current_sentence = []
        current_sentence.append(start_word)
        for i in range(n-2):
            current_word = tuple(current_sentence[-(i+1):])
            next_word = ""
            for element in gram_list[i+1]:
                if current_word == element[0][0:(i+1)]:
                    next_word = element[0][(i+1)]
                    break;
            if next_word == "" :
                break;
            current_sentence.append(next_word)

        '''
        根据前n-1个gram，预测下一个gram，即p(x_m|x_m-1...x_(m-n+1))
        '''
        for i in range(length):
            current_word = tuple(current_sentence[-(n-1):])
            next_word = ""
            for element in gram_list[n-1]:
                if current_word == element[0][0:n-1]:
                    next_word = element[0][n-1]
                    break;
            if next_word == "" :
                break;
            current_sentence.append(next_word)

        print("sentence:",' '.join(current_sentence))