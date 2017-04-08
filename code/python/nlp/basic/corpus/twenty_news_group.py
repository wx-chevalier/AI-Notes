# -*- coding: utf-8 -*-

import fire
import logging
import sys
from collections import defaultdict

import numpy as np

from gensim import matutils
from gensim.models.ldamulticore import LdaMulticore
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from gensim.models.ldamodel import LdaModel



class TwentyNewsGroup(object):
    """
    TwentyNewsGroup 语料集
    """

    def __init__(self):
        self.data = defaultdict(list)
        self.count_vect = CountVectorizer()

    def fetch_data(self, subset='train', categories=None):
        """return data
        执行数据抓取操作

        Arguments:
        subset -> string -- 抓取的目标集合 train / test / all
        """
        rand = np.random.mtrand.RandomState(8675309)
        data = fetch_20newsgroups(subset=subset,
                                  categories=categories,
                                  shuffle=True,
                                  random_state=rand)

        self.data[subset] = data

    def fetch_data_and_dump(self, subset='train', categories=None, output_path='./20newsgroups.txt'):
        """
        执行数据抓取并且将数据持久化存储到磁盘中

        Arguments:
        subset -> string -- 抓取的目标集合 train / test / all
        """

    def extract_feature(self):
        """
        从语料集中抽取文档特征
        """

        # 获取训练数据的文档-词矩阵
        self.train_dtm = self.count_vect.fit_transform(self.data['train'].data)

        # 获取文档的 TF 特征

        tf_transformer = TfidfTransformer(use_idf=False)

        self.train_tf = tf_transformer.transform(self.train_dtm)

        # 获取文档的 TF-IDF 特征

        tfidf_transformer = TfidfTransformer().fit(self.train_dtm)

        self.train_tfidf = tf_transformer.transform(self.train_dtm)

    def train_classifier(self):
        """
        从训练集中训练出分类器
        """

        self.extract_feature();

        self.clf = MultinomialNB().fit(
            self.train_tfidf, self.data['train'].target)

    def predict(self, docs):
        """
        从训练集中训练出分类器
        """

        X_new_counts = self.count_vect.transform(docs)

        tfidf_transformer = TfidfTransformer().fit(X_new_counts)
        
        X_new_tfidf = tfidf_transformer.transform(X_new_counts)

        return self.clf.predict(X_new_tfidf)

    def topics_by_lda(self, num_topics=20, num_words=10):
        """
        利用 LDA 模型进行语料集分析

        Arguments:
        num_topics -> integer -- 既定的主题数目
        num_words -> integer -- 最终返回的单主题词数目
        """

        # 如果是从命令行启动则执行数据抓取
        if not hasattr(self, "data"):
            logging.info("数据集尚未准备，重新准备数据集中！")
            self.fetch_data()

        # 构建语料集统计向量
        vec = CountVectorizer(min_df=10, max_df=80, stop_words='english')

        # 对于数据进行分析
        X = vec.fit_transform(self.data['train'].data)

        # 获取词表
        vocab = vec.get_feature_names()

        # 构建多核 LDA 模型
        lda = LdaModel(
            matutils.Sparse2Corpus(X, documents_columns=False),
            num_topics=num_topics,
            id2word=dict([(i, s) for i, s in enumerate(vocab)])
        )

        # 打印并且返回主题数据
        topics = lda.show_topics(
            num_topics=num_topics,
            num_words=num_words,
            formatted=False,
            log=False)

        for ti, topic in enumerate(topics):
            print("Topic", ti, ":", " ".join(word[0] for word in topic[1]))

        if __name__ != '__main__':
            return topics


if __name__ == '__main__':

    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    fire.Fire(TwentyNewsGroup)
