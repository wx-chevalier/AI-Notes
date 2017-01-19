# -*- coding:utf-8 -*-

import ConfigParser
import numpy as np
import random
import codecs
import os
from collections import OrderedDict

# 获取当前路径
path = os.getcwd()
# 导入配置文件
conf = ConfigParser.ConfigParser()
conf.read("setting.conf")
# 文件路径
trainfile = os.path.join(path, os.path.normpath(conf.get("filepath", "trainfile")))
wordidmapfile = os.path.join(path, os.path.normpath(conf.get("filepath", "wordidmapfile")))
thetafile = os.path.join(path, os.path.normpath(conf.get("filepath", "thetafile")))
phifile = os.path.join(path, os.path.normpath(conf.get("filepath", "phifile")))
paramfile = os.path.join(path, os.path.normpath(conf.get("filepath", "paramfile")))
topNfile = os.path.join(path, os.path.normpath(conf.get("filepath", "topNfile")))
tassginfile = os.path.join(path, os.path.normpath(conf.get("filepath", "tassginfile")))


# LDA模型类
class LDAModel(object):
    def __init__(self, dpre):

        '''
        :function 初始化构造函数
        :param dpre: 预处理之后的数据
        '''

        self.dpre = dpre  # 获取预处理参数

        # 模型参数
        # 聚类个数
        self.K = 3
        # 先验参数
        self.beta = 0.1
        # 先验参数
        self.alpha = 0.1
        # 迭代次数
        self.iter_times = 100
        # 每个类特征词个数
        self.top_words_num = 20

        # 文件变量
        # 分好词的待训练文件
        self.trainfile = trainfile

        # 词对应id文件wordidmapfile
        self.wordidmapfile = wordidmapfile

        # 文章-主题分布文件thetafile
        self.thetafile = thetafile

        # 词-主题分布文件phifile
        self.phifile = phifile

        # 每个主题topN词文件topNfile
        self.topNfile = topNfile

        # 最后分派结果文件tassginfile
        self.tassginfile = tassginfile

        # 模型训练选择的参数文件paramfile
        self.paramfile = paramfile

        # 迭代中间变量

        # p,概率向量 double类型，存储采样的临时变量,尺寸为1 * K
        self.p = np.zeros(self.K)

        # nw,每个词属于每个主题的频次,尺寸为 WordsCount * K
        self.nw = np.zeros((self.dpre.words_count, self.K), dtype="int")

        # nwsum,每个topic的词的总数, 尺寸为 1 * K
        self.nwsum = np.zeros(self.K, dtype="int")

        # nd,每个doc中每个topic的词的总数,尺寸为 DocsCount * K
        self.nd = np.zeros((self.dpre.docs_count, self.K), dtype="int")

        # ndsum,每个doc中词的总数,尺寸为 1 * DocsCount
        self.ndsum = np.zeros(dpre.docs_count, dtype="int")

        # 文档中词的主题分布,尺寸为DocsCount * WordsInDocCount
        self.Z = np.array(
            [
                # 当文档中每个词属于某个主题设置为0
                [0 for y in xrange(dpre.docs[x].length)]

                # 对于每个文档而言
                for x in xrange(dpre.docs_count)

                ])

        # 随机先分配类型
        # 首先遍历每个文档
        for x in xrange(len(self.Z)):
            # 每个文档中词的总数就等于该文档的长度
            self.ndsum[x] = self.dpre.docs[x].length

            # 遍历文档中的每一个词
            for y in xrange(self.dpre.docs[x].length):
                # 随机指定一个Topic
                topic = random.randint(0, self.K - 1)

                # 将该Topic分配给该词
                self.Z[x][y] = topic

                # 将全局状态下该词属于该主题的频次加1
                self.nw[self.dpre.docs[x].words[y]][topic] += 1

                # 将该文档中该Topic的频次加1
                self.nd[x][topic] += 1

                # 将该Topic拥有的词的数目加1
                self.nwsum[topic] += 1

        # 每个文档的主题分布,尺寸为 DocsCount * K
        self.theta = np.array([[0.0 for y in xrange(self.K)] for x in xrange(self.dpre.docs_count)])

        # 每个主题的词分布,尺寸为 K*WordsCount
        self.phi = np.array([[0.0 for y in xrange(self.dpre.words_count)] for x in xrange(self.K)])

    def est(self):
        '''
        :function 开始推断模型参数
        :return: null
        '''
        print (u"迭代次数为%s 次" % self.iter_times)

        # 开始进行总共iter_times次的遍历
        for x in xrange(self.iter_times):
            # 每次遍历中需要遍历所有的文档
            for i in xrange(self.dpre.docs_count):
                # 每个文档中需要遍历所有的词汇
                for j in xrange(self.dpre.docs[i].length):
                    # 遍历每个词时,进行一次采样
                    topic = self.sampling(i, j)

                    # 采样完毕后,将该词对应的主题进行修正
                    self.Z[i][j] = topic

        print (u"迭代完成。")
        print (u"计算文章-主题分布")
        self._theta()
        print (u"计算词-主题分布")
        self._phi()
        print (u"保存模型")
        self.save()

    def sampling(self, i, j):
        '''
        :function 取样函数
        :param i: 第 i 个文档
        :param j: 第 j 个词
        :return: 返回新采样的主题
        '''

        # 获取当前词汇被分配的主题
        topic = self.Z[i][j]

        # 获取当前词汇在词汇表中的编号
        word = self.dpre.docs[i].words[j]

        # 将全局中该词属于该主题的频次减1
        self.nw[word][topic] -= 1

        # 将该文档中该主题的频次减1
        self.nd[i][topic] -= 1

        # 将该主题中的词汇数减1
        self.nwsum[topic] -= 1

        # 将该文档中词汇总数减1
        self.ndsum[i] -= 1

        # 主题在词上的分布
        Vbeta = self.dpre.words_count * self.beta

        # 文档在主题上的分布
        Kalpha = self.K * self.alpha

        # p 的 尺寸为 1 * K
        # [ (该词属于每个主题的频次:1 * K + beta:1 * 1 ) => 1 * K / ( 每个Topic中词的总数:1 * K + VBeta:1 * 1 ) => 1 * K ] => 1 * K
        # *
        # [(该文档中每个Topic的词的总数:1 * K + alpha:1 * 1) => 1 * K / (该文档中词的总数:1 * 1 + Kalpha: 1 * 1) => 1 * 1] => 1 * K
        # => 1 * K
        self.p = (self.nw[word] + self.beta) / (self.nwsum + Vbeta) * \
                 (self.nd[i] + self.alpha) / (self.ndsum[i] + Kalpha)

        # 遍历所有的主题
        for k in xrange(1, self.K):
            # 每个主题设置为后一个值加
            self.p[k] += self.p[k - 1]

        # 设定随机的阈值,使得其至少返回一个主题
        u = random.uniform(0, self.p[self.K - 1])

        # 遍历所有的主题
        for topic in xrange(self.K):

            # 寻找到第一个大于阈值的主题
            if self.p[topic] > u:
                break

        # 将全局中该词属于该主题的频次加1
        self.nw[word][topic] += 1

        # 将该主题中的词汇数加1
        self.nwsum[topic] += 1

        # 将该文档中该主题的频次加1
        self.nd[i][topic] += 1

        # 将该文档中词汇总数加1
        self.ndsum[i] += 1

        # 返回采样取得的主题
        return topic

    def _theta(self):
        '''
        :function 计算每个文档的主题分布
        :return:
        '''

        # 遍历所有文档
        for i in xrange(self.dpre.docs_count):
            # 该文档中每个主题词的数目 / 该文档词的总数目 为 该文档的主题概率分布
            self.theta[i] = (self.nd[i] + self.alpha) / (self.ndsum[i] + self.K * self.alpha)

    def _phi(self):
        '''
        :function 计算每个主题的词分布
        :return:
        '''

        # 遍历所有的主题
        for i in xrange(self.K):
            # 该主题中每个词出现的频次 / (该主题中词的总数 + 词的总数 * beta) 为 该主题中词的概率分布
            self.phi[i] = (self.nw.T[i] + self.beta) / (self.nwsum[i] + self.dpre.words_count * self.beta)

    def save(self):
        '''
        :function 将生成结果保存到文件中
        :return:
        '''

        # 保存theta文章-主题分布
        print (u"文章-主题分布已保存到%s" % self.thetafile)
        with codecs.open(self.thetafile, 'w') as f:
            for x in xrange(self.dpre.docs_count):
                for y in xrange(self.K):
                    f.write(str(self.theta[x][y]) + '\t')
                f.write('\n')
        # 保存phi词-主题分布
        print (u"词-主题分布已保存到%s" % self.phifile)
        with codecs.open(self.phifile, 'w') as f:
            for x in xrange(self.K):
                for y in xrange(self.dpre.words_count):
                    f.write(str(self.phi[x][y]) + '\t')
                f.write('\n')
        # 保存参数设置
        print (u"参数设置已保存到%s" % self.paramfile)
        with codecs.open(self.paramfile, 'w', 'utf-8') as f:
            f.write('K=' + str(self.K) + '\n')
            f.write('alpha=' + str(self.alpha) + '\n')
            f.write('beta=' + str(self.beta) + '\n')
            f.write(u'迭代次数  iter_times=' + str(self.iter_times) + '\n')
            f.write(u'每个类的高频词显示个数  top_words_num=' + str(self.top_words_num) + '\n')
        # 保存每个主题topic的词
        print (u"主题topN词已保存到%s" % self.topNfile)

        with codecs.open(self.topNfile, 'w', 'utf-8') as f:
            self.top_words_num = min(self.top_words_num, self.dpre.words_count)
            for x in xrange(self.K):
                f.write(u'第' + str(x) + u'类：' + '\n')
                twords = []
                twords = [(n, self.phi[x][n]) for n in xrange(self.dpre.words_count)]
                twords.sort(key=lambda i: i[1], reverse=True)
                for y in xrange(self.top_words_num):
                    word = OrderedDict({value: key for key, value in self.dpre.word2id.items()})[twords[y][0]]
                    f.write('\t' * 2 + word + '\t' + str(twords[y][1]) + '\n')
        # 保存最后退出时，文章的词分派的主题的结果
        print(u"文章-词-主题分派结果已保存到%s" % self.tassginfile)
        with codecs.open(self.tassginfile, 'w') as f:
            for x in xrange(self.dpre.docs_count):
                for y in xrange(self.dpre.docs[x].length):
                    f.write(str(self.dpre.docs[x].words[y]) + ':' + str(self.Z[x][y]) + '\t')
                f.write('\n')
        print (u"模型训练完成。")
