# -*- coding: utf-8 -*-

import fire
import logging
import sys

import jieba
from gensim.corpora import WikiCorpus
from gensim.models import word2vec

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class Wiki(object):
    """
    维基百科语料集处理
    """
    
    def wiki2texts(self, wiki_data_path, wiki_texts_path='./wiki_texts.txt'):
        """
        将维基百科数据转化为文本数据

        Arguments:
        wiki_data_path -- 维基压缩文件地址
        """
        if not wiki_data_path:
            print("请输入 Wiki 压缩文件路径或者前往 https://dumps.wikimedia.org/zhwiki/ 下载")
            exit()

        # 构建维基语料集
        wiki_corpus = WikiCorpus(wiki_data_path, dictionary={})
        texts_num = 0

        with open(wiki_text_path, 'w', encoding='utf-8') as output:
            for text in wiki_corpus.get_texts():
                output.write(b' '.join(text).decode('utf-8') + '\n')
                texts_num += 1
                if texts_num % 10000 == 0:
                    logging.info("已处理 %d 篇文章" % texts_num)

        print("处理完毕，请使用 OpenCC 转化为简体字")