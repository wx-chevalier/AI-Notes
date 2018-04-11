# Word2Vec

词向量最直观的理解就是将每一个单词表征为

深度学习（DeepLearning）在图像、语音、视频等多方应用中大放异彩，从本质而言，深度学习是表征学习（Representation Learning）的一种方法，可以看做对事物进行分类的不同过滤器的组成。

Word2Vec 是 Google 在 2013 年年中开源的一款将词表征为实数值向量的高效 工具，采用的模型有 CBOW (Continuous Bag-Of-Words，即连续的词袋模型）和 Skip-Gram 两种。word2vec 代码链接为：https://code.google.com/p/word2vec/， 遵循 Apache License 2.0 开源协议，是一种对商业应用友好的许可，当然需要充分尊重原作者的著作权。Word2Vec 采用了所谓的 Distributed Representation 方式来表示词。Distributed representation 最早是 Hinton 在 1986 年的论文《Learning distributed representations of concepts》中提出的。虽然这篇文章没有说要将词做 Distributed representation，但至少这种先进的思想在那个时候就在人们的心中埋下了火种，到 2000 年之后开始逐渐被人重视。Distributed representation 用来表示词，通常被称为“Word Representation”或“Word Embedding”，中文俗称“词向量”。

![](http://deeplearning4j.org/img/word2vec.png)

Word2vec 是一个神经网络，它用来在使用深度学习算法之前预处理文本。它本身并没有实现深度学习，但是 Word2Vec 把文本变成深度学习能够理解的向量形式。

Word2vec 在不需要人工干预的情况下创建特征，包括词的上下文特征。这些上下文来自于多个词的窗口。如果有足够多的数据，用法和上下文，Word2Vec 能够基于这个词的出现情况高度精确的预测一个词的词义（对于深度学习来说，一个词的词义只是一个简单的信号，这个信号能用来对更大的实体分类；比如把一个文档分类到一个类别中）。

Word2vec 需要一串句子做为其输入。每个句子，也就是一个词的数组，被转换成 n 维向量空间中的一个向量并且可以和其它句子（词的数组）所转换成向量进行比较。在这个向量空间里，相关的词语和词组会出现在一起。把它们变成向量之后，我们可以一定程度的计算它们的相似度并且对其进行聚类。这些类别可以作为搜索，情感分析和推荐的基础。

Word2vec 神经网络的输出是一个词表，每个词由一个向量来表示，这个向量可以做为深度神经网络的输入来进行分类。

## Distributed Representation

> * [Deep-Learning-What-is-meant-by-a-distributed-representation](https://www.quora.com/Deep-Learning/Deep-Learning-What-is-meant-by-a-distributed-representation)

## Reference

### Tutorials & Docs

* [Google - Word2Vec](https://code.google.com/p/word2vec/)
* [Deep Learning 实战之 word2vec](http://techblog.youdao.com/?p=915#LinkTarget_699)
* [word2vector 学习笔记（一）](http://blog.csdn.net/lingerlanlan/article/details/38048335)
* [词向量和语言模型](http://licstar.net/archives/328#s20)

### Practice

* [关于多个词向量算法的实现对比](https://github.com/licstar/compare)
* [斯坦福深度学习课程第二弹：词向量内部和外部任务评价](https://zhuanlan.zhihu.com/p/21391710)

# Quick Start

## Python

笔者推荐使用 Anaconda 这个 Python 的机器学习发布包，此处用的测试数据来自于[这里](http://mattmahoney.net/dc/text8.zip)

* Installation

使用`pip install word2vec`，然后使用`import word2vec`引入

* 文本文件预处理

```
word2vec.word2phrase('/Users/drodriguez/Downloads/text8', '/Users/drodriguez/Downloads/text8-phrases', verbose=True)
```

```
[u'word2phrase', u'-train', u'/Users/drodriguez/Downloads/text8', u'-output', u'/Users/drodriguez/Downloads/text8-phrases', u'-min-count', u'5', u'-threshold', u'100', u'-debug', u'2']
Starting training using file /Users/drodriguez/Downloads/text8
Words processed: 17000K     Vocab size: 4399K  
Vocab size (unigrams + bigrams): 2419827
Words in train file: 17005206
```

### 中文实验

* 语料

  首先准备数据：采用网上博客上推荐的全网新闻数据(SogouCA)，大小为 2.1G。

        从ftp上下载数据包SogouCA.tar.gz：

```
1 wget ftp://ftp.labs.sogou.com/Data/SogouCA/SogouCA.tar.gz --ftp-user=hebin_hit@foxmail.com --ftp-password=4FqLSYdNcrDXvNDi -r
```

          解压数据包：

```
1 gzip -d SogouCA.tar.gz
2 tar -xvf SogouCA.tar
```

          再将生成的txt文件归并到SogouCA.txt中，取出其中包含content的行并转码，得到语料corpus.txt，大小为2.7G。

```
1 cat *.txt > SogouCA.txt
2 cat SogouCA.txt | iconv -f gbk -t utf-8 -c | grep "<content>" > corpus.txt
```

* 分词

  用 ANSJ 对 corpus.txt 进行分词，得到分词结果 resultbig.txt，大小为 3.1G。在分词工具 seg_tool 目录下先编译再执行得到分词结果 resultbig.txt，内含 426221 个词，次数总计 572308385 个。

![](http://img2.tuicool.com/3MNzmu.jpg%21web)

* 词向量训练

```shell
nohup ./word2vec -train resultbig.txt -output vectors.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -sample 1e-3 -threads 12 -binary 1 &
```

* 分析

（1）相似词计算

```
./distance vectors.bin
```

     ./distance可以看成计算词与词之间的距离，把词看成向量空间上的一个点，distance看成向量空间上点与点的距离。  

![](http://img2.tuicool.com/vmYBrq.png!web)

（2）潜在的语言学规律

      在对demo-analogy.sh修改后得到下面几个例子：

      法国的首都是巴黎，英国的首都是伦敦， vector("法国") - vector("巴黎) + vector("英国") --> vector("伦敦")"

![](http://img0.tuicool.com/FrmE73.png%21web)

（3）聚类

    将经过分词后的语料resultbig.txt中的词聚类并按照类别排序：  

```shell
1 nohup ./word2vec -train resultbig.txt -output classes.txt -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -sample 1e-3 -threads 12 -classes 500  &
2 sort classes.txt -k 2 -n > classes_sorted_sogouca.txt
```

![](http://img1.tuicool.com/j6FrAn.png%21web)

（4）短语分析

    先利用经过分词的语料resultbig.txt中得出包含词和短语的文件sogouca_phrase.txt，再训练该文件中词与短语的向量表示。  

```
1 ./word2phrase -train resultbig.txt -output sogouca_phrase.txt -threshold 500 -debug 2
2 ./word2vec -train sogouca_phrase.txt -output vectors_sogouca_phrase.bin -cbow 0 -size 300 -window 10 -negative 0 -hs 1 -sample 1e-3 -threads 12 -binary 1
```

![](http://img2.tuicool.com/7R3Qzq.png%21web)

![](http://img2.tuicool.com/Yjmmmu.png%21web)

## 维基百科实验

# Algorithms

![](http://deeplearning4j.org/img/word2vec_diagrams.png)

## CBOW

CBOW 是 Continuous Bag-of-Words Model 的缩写，是一种与前向 NNLM 类似 的模型，不同点在于 CBOW 去掉了最耗时的非线性隐层且所有词共享隐层。如 下图所示。可以看出，CBOW 模型是预测$P(w*t|w*{t-k},w*{t-(k-1)},\dots,w*{t-1},w*{t+1},\dots,w*{t+k})$。

![](http://7xlgth.com1.z0.glb.clouddn.com/1424C789-5B58-43BA-952C-EACDF43E2AEB.png)

从输入层到隐层所进行的操作实际就是上下文向量的加和，具体的代码如下。 其中 sentence_position 为当前 word 在句子中的下标。以一个具体的句子 A B C D 为例，第一次进入到下面代码时当前 word 为 A，sentence_position 为 0。b 是一 个随机生成的 0 到$window-1$的词，整个窗口的大小为$2*window + 1 - 2*b$，相当于左右各看$window-b$个词。可以看出随着窗口的从左往右滑动，其大小也 是随机的$3 (b=window-1)$到$2\*window+1(b=0)$之间随机变通，即随机值 b 的大小决定了当前窗口的大小。代码中的 neu1 即为隐层向量，也就是上下文（窗口 内除自己之外的词）对应 vector 之和。

![](http://7xlgth.com1.z0.glb.clouddn.com/36F89DA8-F3A0-4C6C-84F8-C31BB19CEEC1.png)

## Skip-Gram

![](http://7xlgth.com1.z0.glb.clouddn.com/F0E76FE8-7B78-4E4C-BB6A-8FB47A67645C.png)

Skip-Gram 模型的图与 CBOW 正好方向相反，从图中看应该 Skip-Gram 应该预测概率$p(w_i,|w_t)$，其中$t - c \le i \le t + c$且$i \ne t,c$是决定上下文窗口大小的常数，$c$越大则需要考虑的 pair 就越多，一般能够带来更精确的结果，但是训练时间也 会增加。假设存在一个$w_1,w_2,w_3,…,w_T$的词组序列，Skip-gram 的目标是最大化：

$$
\frac{1}{T}\sum^{T}_{t=1}\sum_{-c \le j \le c, j \ne 0}log p(w\_{t+j}|w_t)
$$

基本的 Skip-Gram 模型定义$p(w_o|w_I)$为：

$$
P(w*o | w_I) = \frac{e^{v*{w*o}^{T*{V*{w_I}}}}}{\Sigma*{w=1}^{W}e^{V*w^{T*{V\_{w_I}}}}}
$$

从公式不难看出，Skip-Gram 是一个对称的模型，如果$w_t$为中心词时$w_k$在其窗口内，则$w_t$也必然在以$w_k$为中心词的同样大小的窗口内，也就是：

$$
\frac{1}{T}\sum^{T}_{t=1}\sum_{-c \le j \le c, j \ne 0}log p(w*{t+j}|w_t) = \\ \frac{1}{T}\sum^{T}*{t=1}\sum*{-c \le j \le c, j \ne 0}log p(w*{t}|w\_{t+j})
$$

同时，Skip-Gram 中的每个词向量表征了上下文的分布。Skip-Gram 中的 Skip 是指在一定窗口内的词两两都会计算概率，就算他们之间隔着一些词，这样的好处是“白色汽车”和“白色的汽车”很容易被识别为相同的短语。

与 CBOW 类似，Skip-Gram 也有两种可选的算法：层次 Softmax 和 Negative Sampling。层次 Sofamax 算法也结合了 Huffman 编码，每个词$w$都可以从树的根节点沿着唯一一条路径被访问到。假设$n(w,j)$为这条路径上的第$j$个结点，且$L(w)$为这条路径的长度，注意$j$从 1 开始编码，即$n(w,1)=root,n(w,L(w))=w$。层次 Softmax 定义的概率$p(w|w_I)$为：

$$
p(w|w*I)=\Pi*{j=1}^{L(w)-1}\sigma([n(w,j+1)=ch(n(w,j))]\*v'^T\_{n(w,j)}v_I)
$$

$ch(n(w,j))$既可以是$n(w,j)$的左子结点也可以是$n(w,j)$的右子结点，word2vec 源代码中采用的是左子节点(Label 为$1-code[j]$)，其实此处改为右子节点也是可以的。

# Tricks

## Learning Phrases

对于某些词语，经常出现在一起的，我们就判定他们是短语。那么如何衡量呢？用以下公式。

$score(w_i,w_j)=\frac{count(w_iw_j) - \delta}{count(w_i) \* count(w_j)}$

输入两个词向量，如果算出的 score 大于某个阈值时，我们就认定他们是“在一起的”。为了考虑到更长的短语，我们拿 2-4 个词语作为训练数据，依次降低阈值。

# Implementation

Word2Vec 高效率的原因可以认为如下：

1.去掉了费时的非线性隐层；

2.Huffman Huffman 编码 相当于做了一定聚类 ，不需要统计所有词对 ；

3.Negative Sampling；

4.随机梯度算法；

5.只过一遍数据 ，不需要反复迭代 ；

6.编程实现中的一些 trick，比如指数运算的预计，高频词亚采样等 。

word2vec 可调整的超参数有很多：

| 参数名     | 说明                 |                                                                                                    |
| ---------- | -------------------- | -------------------------------------------------------------------------------------------------- |
| -size      | 向量维度             | 一般维度越高越好，但并不总是这样。                                                                 |
| -window    | 上下文窗口大小       | Skip-gram—般 10 左右，CBOW—般 5 左右。                                                             |
| -sample    | 高频词亚采样         | 对大数据集合可以同时提高精度和速度，sample 的取值 在 1e-3 到 1e-5 之间效果最佳。                   |
| -hs        | 是否采用层次 softmax | 层次 softmax 对低频词效果更好；对应的 negative sampling 对高频词效果更好，向量维度较低时效果更好。 |
| -negative  | 负例数目             |                                                                                                    |
| -min-count | 被截断的低频词阈值   |                                                                                                    |
| -alpha     | 开始的学习速率       |                                                                                                    |
| -cbow      | 使用 CBOW            | Skip-gram 更慢一些，但是对低频词效果更好；对应的 CBOW 则速度更快一些。                             |

## Deeplearning4j

> [Word2vec](http://deeplearning4j.org/zh-word2vec.html)
>
> [DL4J-Word2Vec](http://deeplearning4j.org/word2vec.html#intro)

## Python

> [中英文维基百科语料上的 Word2Vec 实验](http://www.52nlp.cn/%E4%B8%AD%E8%8B%B1%E6%96%87%E7%BB%B4%E5%9F%BA%E7%99%BE%E7%A7%91%E8%AF%AD%E6%96%99%E4%B8%8A%E7%9A%84word2vec%E5%AE%9E%E9%AA%8C)

```
%load_ext autoreload
%autoreload 2
```

# word2vec

This notebook is equivalent to `demo-word.sh`, `demo-analogy.sh`, `demo-phrases.sh` and `demo-classes.sh` from Google.

## Training

Download some data, for example: [http://mattmahoney.net/dc/text8.zip](http://mattmahoney.net/dc/text8.zip)

```
import word2vec
```

Run `word2phrase` to group up similar words "Los Angeles" to "Los_Angeles"

```
word2vec.word2phrase('/Users/drodriguez/Downloads/text8', '/Users/drodriguez/Downloads/text8-phrases', verbose=True)
```

```
[u'word2phrase', u'-train', u'/Users/drodriguez/Downloads/text8', u'-output', u'/Users/drodriguez/Downloads/text8-phrases', u'-min-count', u'5', u'-threshold', u'100', u'-debug', u'2']
Starting training using file /Users/drodriguez/Downloads/text8
Words processed: 17000K     Vocab size: 4399K  
Vocab size (unigrams + bigrams): 2419827
Words in train file: 17005206
```

This will create a `text8-phrases` that we can use as a better input for `word2vec`.Note that you could easily skip this previous step and use the origial data as input for `word2vec`.

Train the model using the `word2phrase` output.

```
word2vec.word2vec('/Users/drodriguez/Downloads/text8-phrases', '/Users/drodriguez/Downloads/text8.bin', size=100, verbose=True)
```

```
Starting training using file /Users/drodriguez/Downloads/text8-phrases
Vocab size: 98331
Words in train file: 15857306
Alpha: 0.000002  Progress: 100.03%  Words/thread/sec: 286.52k  
```

That generated a `text8.bin` file containing the word vectors in a binary format.

Do the clustering of the vectors based on the trained model.

```
word2vec.word2clusters('/Users/drodriguez/Downloads/text8', '/Users/drodriguez/Downloads/text8-clusters.txt', 100, verbose=True)
```

```
Starting training using file /Users/drodriguez/Downloads/text8
Vocab size: 71291
Words in train file: 16718843
Alpha: 0.000002  Progress: 100.02%  Words/thread/sec: 287.55k  
```

That created a `text8-clusters.txt` with the cluster for every word in the vocabulary

## Predictions

```
import word2vec
```

Import the `word2vec` binary file created above

```
model = word2vec.load('/Users/drodriguez/Downloads/text8.bin')
```

We can take a look at the vocabulaty as a numpy array

```
model.vocab
```

```
array([u'</s>', u'the', u'of', ..., u'dakotas', u'nias', u'burlesques'],
      dtype='<U78')
```

Or take a look at the whole matrix

```
model.vectors.shape
```

```
(98331, 100)
```

```
model.vectors
```

```
array([[ 0.14333282,  0.15825513, -0.13715845, ...,  0.05456942,
         0.10955409,  0.00693387],
       [ 0.1220774 ,  0.04939618,  0.09545057, ..., -0.00804222,
        -0.05441621, -0.10076696],
       [ 0.16844609,  0.03734054,  0.22085373, ...,  0.05854521,
         0.04685341,  0.02546694],
       ...,
       [-0.06760896,  0.03737842,  0.09344187, ...,  0.14559349,
        -0.11704484, -0.05246212],
       [ 0.02228479, -0.07340827,  0.15247506, ...,  0.01872172,
        -0.18154132, -0.06813737],
       [ 0.02778879, -0.06457976,  0.07102411, ..., -0.00270281,
        -0.0471223 , -0.135444  ]])
```

We can retreive the vector of individual words

```
model['dog'].shape
```

```
(100,)
```

```
model['dog'][:10]
```

```
array([ 0.05753701,  0.0585594 ,  0.11341395,  0.02016246,  0.11514406,
        0.01246986,  0.00801256,  0.17529851,  0.02899276,  0.0203866 ])
```

We can do simple queries to retreive words similar to "socks" based on cosine similarity:

```
indexes, metrics = model.cosine('socks')
indexes, metrics
```

```
(array([20002, 28915, 30711, 33874, 27482, 14631, 22992, 24195, 25857, 23705]),
 array([ 0.8375354 ,  0.83590846,  0.82818749,  0.82533614,  0.82278399,
         0.81476386,  0.8139092 ,  0.81253798,  0.8105933 ,  0.80850171]))
```

This returned a tuple with 2 items:

1. numpy array with the indexes of the similar words in the vocabulary
2. numpy array with cosine similarity to each word

Its possible to get the words of those indexes

```
model.vocab[indexes]
```

```
array([u'hairy', u'pumpkin', u'gravy', u'nosed', u'plum', u'winged',
       u'bock', u'petals', u'biscuits', u'striped'],
      dtype='<U78')
```

There is a helper function to create a combined response: a numpy [record array](http://docs.scipy.org/doc/numpy/user/basics.rec.html)

```
model.generate_response(indexes, metrics)
```

```
rec.array([(u'hairy', 0.8375353970603848), (u'pumpkin', 0.8359084628493809),
       (u'gravy', 0.8281874915608026), (u'nosed', 0.8253361379785071),
       (u'plum', 0.8227839904046932), (u'winged', 0.8147638561412592),
       (u'bock', 0.8139092031538545), (u'petals', 0.8125379796045767),
       (u'biscuits', 0.8105933044655644), (u'striped', 0.8085017054444408)],
      dtype=[(u'word', '<U78'), (u'metric', '<f8')])
```

Is easy to make that numpy array a pure python response:

```
model.generate_response(indexes, metrics).tolist()
```

```
[(u'hairy', 0.8375353970603848),
 (u'pumpkin', 0.8359084628493809),
 (u'gravy', 0.8281874915608026),
 (u'nosed', 0.8253361379785071),
 (u'plum', 0.8227839904046932),
 (u'winged', 0.8147638561412592),
 (u'bock', 0.8139092031538545),
 (u'petals', 0.8125379796045767),
 (u'biscuits', 0.8105933044655644),
 (u'striped', 0.8085017054444408)]
```

### Phrases

Since we trained the model with the output of `word2phrase` we can ask for similarity of "phrases"

```
indexes, metrics = model.cosine('los_angeles')
model.generate_response(indexes, metrics).tolist()
```

```
[(u'san_francisco', 0.886558000570455),
 (u'san_diego', 0.8731961018831669),
 (u'seattle', 0.8455603712285231),
 (u'las_vegas', 0.8407843553947962),
 (u'miami', 0.8341796009062884),
 (u'detroit', 0.8235412519780195),
 (u'cincinnati', 0.8199138493085706),
 (u'st_louis', 0.8160655356728751),
 (u'chicago', 0.8156786240847214),
 (u'california', 0.8154244925085712)]
```

### Analogies

Its possible to do more complex queries like analogies such as: `king - man + woman = queen` This method returns the same as `cosine` the indexes of the words in the vocab and the metric

```
indexes, metrics = model.analogy(pos=['king', 'woman'], neg=['man'], n=10)
indexes, metrics
```

```
(array([1087, 1145, 7523, 3141, 6768, 1335, 8419, 1826,  648, 1426]),
 array([ 0.2917969 ,  0.27353295,  0.26877692,  0.26596514,  0.26487509,
         0.26428581,  0.26315492,  0.26261258,  0.26136635,  0.26099078]))
```

```
model.generate_response(indexes, metrics).tolist()
```

```
[(u'queen', 0.2917968955611075),
 (u'prince', 0.27353295205311695),
 (u'empress', 0.2687769174818083),
 (u'monarch', 0.2659651399832089),
 (u'regent', 0.26487508713026797),
 (u'wife', 0.2642858109968327),
 (u'aragon', 0.2631549214361766),
 (u'throne', 0.26261257728511833),
 (u'emperor', 0.2613663460665488),
 (u'bishop', 0.26099078142148696)]
```

### Clusters

```
clusters = word2vec.load_clusters('/Users/drodriguez/Downloads/text8-clusters.txt')
```

We can see get the cluster number for individual words

```
clusters['dog']
```

```
11
```

We can see get all the words grouped on an specific cluster

```
clusters.get_words_on_cluster(90).shape
```

```
(221,)
```

```
clusters.get_words_on_cluster(90)[:10]
```

```
array(['along', 'together', 'associated', 'relationship', 'deal',
       'combined', 'contact', 'connection', 'bond', 'respect'], dtype=object)
```

We can add the clusters to the word2vec model and generate a response that includes the clusters

```
model.clusters = clusters
```

```
indexes, metrics = model.analogy(pos=['paris', 'germany'], neg=['france'], n=10)
```

```
model.generate_response(indexes, metrics).tolist()
```

```
[(u'berlin', 0.32333651414395953, 20),
 (u'munich', 0.28851564633559, 20),
 (u'vienna', 0.2768927258877336, 12),
 (u'leipzig', 0.2690537010929304, 91),
 (u'moscow', 0.26531859560322785, 74),
 (u'st_petersburg', 0.259534503067277, 61),
 (u'prague', 0.25000637367753303, 72),
 (u'dresden', 0.2495974800117785, 71),
 (u'bonn', 0.24403155303236473, 8),
 (u'frankfurt', 0.24199720792200027, 31)]
```

```

```
