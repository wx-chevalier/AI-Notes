# -*- coding: utf-8 -*-

import fire
import logging
import sys

import jieba
from gensim.corpora import WikiCorpus
from gensim.models import word2vec

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from text_processor import TextProcessor

# 设置显示中文字
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 结巴分词字典地址
jieba_dictionary = '/Users/apple/Workspace/Data/jieba/dict.big.txt'

# 结巴分词停止词地址
jieba_stopwords = '/Users/apple/Workspace/Data/jieba/stopwords.txt'


class ChineseTextProcessor(TextProcessor):
    """
    常见中文语料集处理
    """

    def __init__(self):
        """
        构造函数
        """

    def __config_jieba(self):
        """
        配置结巴分词
        """
        jieba.set_dictionary(jieba_dictionary)

    def tokenize_file(self, text_path, text_output_path='./tokenized_texts.txt'):
        """
        将指定的文本利用 jieba 进行分词
        """

        # jieba custom setting.
        jieba.set_dictionary(jieba_dictionary)

        # load stopwords set
        stopwordset = set()

        with open(jieba_stopwords, 'r', encoding='utf-8') as sw:
            for line in sw:
                stopwordset.add(line.strip('\n'))

        # 统计
        texts_num = 0

        # 打开输出文件
        output = open(text_output_path, 'w')

        # 遍历所有的行
        with open(text_path, 'r') as content:
            for line in content:
                line = line.strip('\n')

                # 进行分词操作
                words = jieba.cut(line, cut_all=False)
                for word in words:
                    if word not in stopwordset:
                        output.write(word + ' ')

                output.write('\n')

                texts_num += 1
                if texts_num % 10000 == 0:
                    logging.info("已完成前 %d 行的分词" % texts_num)
        output.close()


if __name__ == '__main__':

    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    fire.Fire(ChineseTextProcessor)
