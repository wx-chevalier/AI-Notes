# -*- coding: utf-8 -*-
import fire
import logging
import sys

from gensim.corpora import WikiCorpus
from gensim.models import word2vec

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from config import ENGLISH_STOP_WORDS


class TextProcessor(object):
    """
    常见文本处理
    """

    def wv_train(self, tokenized_text_path, output_model_path='./wv_model.bin'):
        """
        对于文本进行词向量训练，并将输出的词向量保存
        """

        sentences = word2vec.Text8Corpus(tokenized_text_path)

        # 进行模型训练
        model = word2vec.Word2Vec(sentences, size=250)

        # 保存模型
        model.save(output_model_path)

    def wv_visualize(self, model_path, word=["中国", "航空"]):
        """
        根据输入的词搜索邻近词然后可视化展示

        参数：
            model_path: Word2Vec 模型地址
        """

        # 加载模型
        model = word2vec.Word2Vec.load(model_path)

        # 寻找出最相似的多个词
        words = [wp[0] for wp in model.most_similar(word, topn=20)]

        # 提取出词对应的词向量
        wordsInVector = [model[word] for word in words]

        # 进行 PCA 降维
        pca = PCA(n_components=2)
        pca.fit(wordsInVector)
        X = pca.transform(wordsInVector)

        # 绘制图形
        xs = X[:, 0]
        ys = X[:, 1]

        plt.figure(figsize=(12, 8))
        plt.scatter(xs, ys, marker='o')

        # 遍历所有的词添加点注释
        for i, w in enumerate(words):
            plt.annotate(
                w,
                xy=(xs[i], ys[i]), xytext=(6, 6),
                textcoords='offset points', ha='left', va='top',
                **dict(fontsize=10)
            )
        plt.show()


if __name__ == '__main__':

    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    fire.Fire(TextProcessor)
