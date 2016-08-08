# -*- coding:utf-8 -*-

import logging
import logging.config
import ConfigParser
from xml.dom.minidom import Document
import numpy as np
import random
import codecs
import os
from collections import OrderedDict

# 获取当前路径
path = os.getcwd()

# 读取当前配置
conf = ConfigParser.ConfigParser()
conf.read("setting.conf")

# 设置当前的词文件
trainfile = os.path.join(path, os.path.normpath(conf.get("filepath", "trainfile")))
wordidmapfile = os.path.join(path, os.path.normpath(conf.get("filepath", "wordidmapfile")))


class Document(object):
    '''
    :function 文档对象
    '''

    def __init__(self):
        self.words = []
        self.length = 0


# 数据与处理类
class Preprocessor(object):
    def __init__(self):
        # 文档的总数
        self.docs_count = 0

        # 词的总数
        self.words_count = 0

        # 某个文档中词的数目
        self.docs = []

        # 词的编号
        self.word2id = OrderedDict()

    def preprocessing(self):
        '''
        :function 执行数据预处理操作
        :return:
        '''
        print (u'载入数据......')

        # 打开文件读入数据
        with codecs.open(trainfile, 'r', 'utf-8') as f:
            docs = f.readlines()

        print (u"载入完成,准备生成字典对象和统计文本数据...")

        # 记录某个词的下标
        items_idx = 0

        # 读取文档的每一行
        for line in docs:
            if line != "":
                tmp = line.strip().split()
                # 生成一个文档对象
                doc = Document()

                # 遍历所有的词汇
                for item in tmp:

                    # 判断是否已经包含了该词汇
                    if self.word2id.has_key(item):

                        # 如果已经包含则直接添加
                        doc.words.append(self.word2id[item])
                    else:

                        # 否则将该词汇的下标设置为items_idx
                        self.word2id[item] = items_idx

                        # 将该单词下标添加到文档列表
                        doc.words.append(items_idx)

                        # 单词下标加1
                        items_idx += 1

                # 设置文档长度
                doc.length = len(tmp)

                self.docs.append(doc)
            else:
                pass
        self.docs_count = len(self.docs)

        self.words_count = len(self.word2id)

        print (u"共有%s个文档" % self.docs_count)

        self.cachewordidmap()

        print (u"词与序号对应关系已保存到%s" % wordidmapfile)


    def cachewordidmap(self):
        '''
        :function 将数据存储到硬盘中
        :return:
        '''
        with codecs.open(wordidmapfile, 'w', 'utf-8') as f:
            for word, id in self.word2id.items():
                f.write(word + "\t" + str(id) + "\n")
