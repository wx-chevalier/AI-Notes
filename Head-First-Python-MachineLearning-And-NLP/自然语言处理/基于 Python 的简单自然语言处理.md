> [基于 Python 的简单自然语言处理](https://zhuanlan.zhihu.com/p/26249110) 从属于笔者的 [程序猿的数据科学与机器学习实战手册](https://github.com/wxyyxc1992/DataScience-And-MachineLearning-Handbook-For-Coders)。

# 基于 Python 的简单自然语言处理

本文是对于基于 Python 进行简单自然语言处理任务的介绍，本文的所有代码放置在[这里](https://parg.co/b4h)。建议前置阅读 [Python 语法速览与机器学习开发环境搭建](https://zhuanlan.zhihu.com/p/24536868)，更多机器学习资料参考[机器学习、深度学习与自然语言处理领域推荐的书籍列表](https://zhuanlan.zhihu.com/p/25612011)以及[面向程序猿的数据科学与机器学习知识体系及资料合集](https://parg.co/b4C)。

# Twenty News Group 语料集处理

20 Newsgroup 数据集包含了约 20000 篇来自于不同的新闻组的文档，最早由 Ken Lang 搜集整理。本部分包含了对于数据集的抓取、特征提取、简单分类器训练、主题模型训练等。本部分代码包括主要的处理代码[封装库](https://parg.co/b4M)与[基于 Notebook 的交互示范](https://parg.co/b4t)。我们首先需要进行数据抓取：

```
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
```

然后在 Notebook 中交互查看数据格式：

```
# 实例化对象
twp = TwentyNewsGroup()
# 抓取数据
twp.fetch_data()
twenty_train = twp.data['train']
print("数据集结构", "->", twenty_train.keys())
print("文档数目", "->", len(twenty_train.data))
print("目标分类", "->",[ twenty_train.target_names[t] for t in twenty_train.target[:10]])

数据集结构 -> dict_keys(['data', 'filenames', 'target_names', 'target', 'DESCR', 'description'])
文档数目 -> 11314
目标分类 -> ['sci.space', 'comp.sys.mac.hardware', 'sci.electronics', 'comp.sys.mac.hardware', 'sci.space', 'rec.sport.hockey', 'talk.religion.misc', 'sci.med', 'talk.religion.misc', 'talk.politics.guns']
```

接下来我们可以对语料集中的特征进行提取：

```
# 进行特征提取

# 构建文档-词矩阵（Document-Term Matrix）

from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(twenty_train.data)

print("DTM 结构","->",X_train_counts.shape)

# 查看某个词在词表中的下标
print("词对应下标","->", count_vect.vocabulary_.get(u'algorithm'))

DTM 结构 -> (11314, 130107)
词对应下标 -> 27366
```

为了将文档用于进行分类任务，还需要使用 TF-IDF 等常见方法将其转化为特征向量：

```
# 构建文档的 TF 特征向量
from sklearn.feature_extraction.text import TfidfTransformer

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

print("某文档 TF 特征向量","->",X_train_tf)

# 构建文档的 TF-IDF 特征向量
from sklearn.feature_extraction.text import TfidfTransformer

tf_transformer = TfidfTransformer().fit(X_train_counts)
X_train_tfidf = tf_transformer.transform(X_train_counts)

print("某文档 TF-IDF 特征向量","->",X_train_tfidf)

某文档 TF 特征向量 ->   (0, 6447)	0.0380693493813
  (0, 37842)	0.0380693493813
```

我们可以将特征提取、分类器训练与预测封装为单独函数：

```
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
```

然后执行训练并且进行预测与评价：

```
# 训练分类器
twp.train_classifier()

# 执行预测
docs_new = ['God is love', 'OpenGL on the GPU is fast']
predicted = twp.predict(docs_new)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))

# 执行模型评测
twp.fetch_data(subset='test')

predicted = twp.predict(twp.data['test'].data)

import numpy as np

# 误差计算

# 简单误差均值
np.mean(predicted == twp.data['test'].target)

# Metrics

from sklearn import metrics

print(metrics.classification_report(
    twp.data['test'].target, predicted,
    target_names=twp.data['test'].target_names))

# Confusion Matrix
metrics.confusion_matrix(twp.data['test'].target, predicted)

'God is love' => soc.religion.christian
'OpenGL on the GPU is fast' => rec.autos
                          precision    recall  f1-score   support

             alt.atheism       0.79      0.50      0.61       319
           ...
      talk.religion.misc       1.00      0.08      0.15       251

             avg / total       0.82      0.79      0.77      7532

Out[16]:
array([[158,   0,   1,   1,   0,   1,   0,   3,   7,   1,   2,   6,   1,
          8,   3, 114,   6,   7,   0,   0],
       ...
       [ 35,   3,   1,   0,   0,   0,   1,   4,   1,   1,   6,   3,   0,
          6,   5, 127,  30,   5,   2,  21]])
```

我们也可以对文档集进行主题提取：

```
# 进行主题提取

twp.topics_by_lda()

Topic 0 : stream s1 astronaut zoo laurentian maynard s2 gtoal pem fpu
Topic 1 : 145 cx 0d bh sl 75u 6um m6 sy gld
Topic 2 : apartment wpi mars nazis monash palestine ottoman sas winner gerard
Topic 3 : livesey contest satellite tamu mathew orbital wpd marriage solntze pope
Topic 4 : x11 contest lib font string contrib visual xterm ahl brake
Topic 5 : ax g9v b8f a86 1d9 pl 0t wm 34u giz
Topic 6 : printf null char manes behanna senate handgun civilians homicides magpie
Topic 7 : buf jpeg chi tor bos det que uwo pit blah
Topic 8 : oracle di t4 risc nist instruction msg postscript dma convex
Topic 9 : candida cray yeast viking dog venus bloom symptoms observatory roby
Topic 10 : cx ck hz lk mv cramer adl optilink k8 uw
Topic 11 : ripem rsa sandvik w0 bosnia psuvm hudson utk defensive veal
Topic 12 : db espn sabbath br widgets liar davidian urartu sdpa cooling
Topic 13 : ripem dyer ucsu carleton adaptec tires chem alchemy lockheed rsa
Topic 14 : ingr sv alomar jupiter borland het intergraph factory paradox captain
Topic 15 : militia palestinian cpr pts handheld sharks igc apc jake lehigh
Topic 16 : alaska duke col russia uoknor aurora princeton nsmca gene stereo
Topic 17 : uuencode msg helmet eos satan dseg homosexual ics gear pyron
Topic 18 : entries myers x11r4 radar remark cipher maine hamburg senior bontchev
Topic 19 : cubs ufl vitamin temple gsfc mccall astro bellcore uranium wesleyan
```

# 常见自然语言处理工具封装

经过上面对于 20NewsGroup 语料集处理的介绍我们可以发现常见自然语言处理任务包括，数据获取、数据预处理、数据特征提取、分类模型训练、主题模型或者词向量等高级特征提取等等。笔者还习惯用 [python-fire](https://github.com/google/python-fire) 将类快速封装为可通过命令行调用的工具，同时也支持外部模块调用使用。本部分我们主要以中文语料集为例，譬如我们需要对中文维基百科数据进行分析，可以使用 gensim 中的[维基百科处理类](https://parg.co/b44)：

```
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
```

抓取完毕后，我们还需要用 OpenCC 转化为简体字。抓取完毕后我们可以使用结巴分词对生成的文本文件进行分词，代码参考[这里](https://parg.co/b4R)，我们直接使用 `python chinese_text_processor.py tokenize_file /output.txt` 直接执行该任务并且生成输出文件。获取分词之后的文件，我们可以将其转化为简单的词袋表示或者文档-词向量，详细代码参考[这里](https://parg.co/b4f)：

```
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
```

我们也可以对分词之后的文档进行主题模型或者词向量提取，这里使用分词之后的文件就可以忽略中英文的差异：

```
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
```

该函数同样可以使用命令行直接调用，传入分词之后的文件。我们也可以对其语料集建立词向量，代码参考[这里](https://parg.co/b4N)；如果对于词向量基本使用尚不熟悉的同学可以参考[基于 Gensim 的 Word2Vec 实践](https://zhuanlan.zhihu.com/p/24961011)：

```
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
```
