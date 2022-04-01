#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022/3/30 上午11:27
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
import urllib.request
import gzip

class spyder(object):
    def __init__(self):

        books = {'Pride and Prejudice': '1342',
         'Huckleberry Fin': '76',
         'Sherlock Holmes': '1661'}
        self.books = books

        url_template = 'https://www.gutenberg.org/cache/epub/%s/pg%s.txt.utf8.gzip'
        self.url_template = url_template

    def getUrlContent(self,book):
        '''
        根据url抓取内容进行解析
        '''
        url_template = self.url_template
        response = urllib.request.urlopen(url_template % (book, book))
        html = response.read()
        try:
            txt = gzip.decompress(html).decode(encoding="utf-8", errors="ignore")
        except:
            txt = html.decode(encoding="utf-8", errors="ignore")
        response.close()
        return txt

    def crawl(self,book):
        bookid = self.books[book]
        txt = self.getUrlContent(bookid)
        print ("文本总共长：",len(txt), ',前30个字符是：', txt[:30], '...')
        return txt