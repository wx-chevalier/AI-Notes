# -*- coding: utf-8 -*-

import fire
import logging
import sys
import re

from gensim import matutils
from gensim import corpora
from gensim.models.ldamulticore import LdaMulticore
from sklearn.feature_extraction.text import CountVectorizer

# 示例文档
default_documents = [['human', 'interface', 'computer'],
                     ['survey', 'user', 'computer', 'system', 'response', 'time'],
                     ['eps', 'user', 'interface', 'system'],
                     ['system', 'human', 'system', 'eps'],
                     ['user', 'response', 'time'],
                     ['trees'],
                     ['graph', 'trees'],
                     ['graph', 'minors', 'trees'],
                     ['graph', 'minors', 'survey']]


class CorpusProcessor:
    """
    语料集处理
    """

    def corpus2bow(self, tokenized_corpus=default_documents):
        """returns (vocab,corpus_in_bow)

        将语料集转化为 BOW 形式

        Arguments:
        tokenized_corpus -- 经过分词的文档列表

        Return:
        vocab -- {'human': 0, ... 'minors': 11}
        corpus_in_bow -- [[(0, 1), (1, 1), (2, 1)]...]
        """
        dictionary = corpora.Dictionary(tokenized_corpus)

        # 获取词表
        vocab = dictionary.token2id

        # 获取文档的词袋表示
        corpus_in_bow = [dictionary.doc2bow(text) for text in tokenized_corpus]

        return (vocab, corpus_in_bow)

    def corpus2bow_file(self, tokenized_corpus_path, vocab_output_path="./vocab.mm", corpus_output_path="./corpus_in_bow.mm", type="gensim"):
        """
        从磁盘中读入语料集将其转化为 BOW 并且写入到磁盘中

        """

    def corpus2dtm(self, tokenized_corpus=default_documents, min_df=10, max_df=100):
        """returns (vocab, DTM)

        将语料集转化为文档-词矩阵

        - dtm -> matrix: 文档-词矩阵
                I	like	hate	databases
        D1	1	  1	      0	        1
        D2	1	  0	      1	        1

        """

        if type(tokenized_corpus[0]) is list:
            documents = [" ".join(document) for document in tokenized_corpus]
        else:
            documents = tokenized_corpus

        if max_df == -1:
            max_df = round(len(documents) / 2)

        # 构建语料集统计向量
        vec = CountVectorizer(min_df=min_df,
                              max_df=max_df,
                              analyzer="word",
                              token_pattern="[\S]+",
                              tokenizer=None,
                              preprocessor=None,
                              stop_words=None
                              )

        # 对于数据进行分析
        DTM = vec.fit_transform(documents)

        # 获取词表
        vocab = vec.get_feature_names()

        return (vocab, DTM)

    def topics_by_lda(self, tokenized_corpus_path, num_topics=20, num_words=10, max_lines=10000, split="\s+", max_df=100):
        """
        读入经过分词的文件并且对其进行 LDA 训练

        Arguments:
        tokenized_corpus_path -> string -- 经过分词的语料集地址
        num_topics -> integer -- 主题数目
        num_words -> integer -- 主题词数目
        max_lines -> integer -- 每次读入的最大行数
        split -> string -- 文档的词之间的分隔符
        max_df -> integer -- 避免常用词，过滤超过该阈值的词
        """

        # 存放所有语料集信息
        corpus = []

        with open(tokenized_corpus_path, 'r', encoding='utf-8') as tokenized_corpus:

            flag = 0

            for document in tokenized_corpus:

                # 判断是否读取了足够的行数
                if(flag > max_lines):
                    break

                # 将读取到的内容添加到语料集中
                corpus.append(re.split(split, document))

                flag = flag + 1

        # 构建语料集的 BOW 表示
        (vocab, DTM) = self.corpus2dtm(corpus, max_df=max_df)

        # 训练 LDA 模型

        lda = LdaMulticore(
            matutils.Sparse2Corpus(DTM, documents_columns=False),
            num_topics=num_topics,
            id2word=dict([(i, s) for i, s in enumerate(vocab)]),
            workers=4
        )

        # 打印并且返回主题数据
        topics = lda.show_topics(
            num_topics=num_topics,
            num_words=num_words,
            formatted=False,
            log=False)

        for ti, topic in enumerate(topics):
            print("Topic", ti, ":", " ".join(word[0] for word in topic[1]))


if __name__ == '__main__':

    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    fire.Fire(CorpusProcessor)
