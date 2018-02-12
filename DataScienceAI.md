<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Introduction](#introduction)
  - [Reference](#reference)
    - [Tutorials & Docs](#tutorials-&-docs)
      - [Video Courses:视频教程](#video-courses%E8%A7%86%E9%A2%91%E6%95%99%E7%A8%8B)
    - [Practices & Resources](#practices-&-resources)
      - [Collection](#collection)
    - [Blogs & News](#blogs-&-news)
    - [Books & Tools](#books-&-tools)
- [DataScience LifeCycle:一次数据研究的生命周期](#datascience-lifecycle%E4%B8%80%E6%AC%A1%E6%95%B0%E6%8D%AE%E7%A0%94%E7%A9%B6%E7%9A%84%E7%94%9F%E5%91%BD%E5%91%A8%E6%9C%9F)
  - [Identify Problem:定位问题](#identify-problem%E5%AE%9A%E4%BD%8D%E9%97%AE%E9%A2%98)
  - [Data PreProcess:数据预处理](#data-preprocess%E6%95%B0%E6%8D%AE%E9%A2%84%E5%A4%84%E7%90%86)
    - [数据集成、数据冗余与数值冲突](#%E6%95%B0%E6%8D%AE%E9%9B%86%E6%88%90%E3%80%81%E6%95%B0%E6%8D%AE%E5%86%97%E4%BD%99%E4%B8%8E%E6%95%B0%E5%80%BC%E5%86%B2%E7%AA%81)
    - [数据采样](#%E6%95%B0%E6%8D%AE%E9%87%87%E6%A0%B7)
    - [数据清洗、缺失值处理与噪声数据](#%E6%95%B0%E6%8D%AE%E6%B8%85%E6%B4%97%E3%80%81%E7%BC%BA%E5%A4%B1%E5%80%BC%E5%A4%84%E7%90%86%E4%B8%8E%E5%99%AA%E5%A3%B0%E6%95%B0%E6%8D%AE)
  - [Feature Engineering:特征工程](#feature-engineering%E7%89%B9%E5%BE%81%E5%B7%A5%E7%A8%8B)
    - [特征的提取、选择和构造](#%E7%89%B9%E5%BE%81%E7%9A%84%E6%8F%90%E5%8F%96%E3%80%81%E9%80%89%E6%8B%A9%E5%92%8C%E6%9E%84%E9%80%A0)
  - [Model Selection&Evaluation:模型选择与评估](#model-selection&evaluation%E6%A8%A1%E5%9E%8B%E9%80%89%E6%8B%A9%E4%B8%8E%E8%AF%84%E4%BC%B0)
- [History of Machine Learning:机器学习的前世今生](#history-of-machine-learning%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%9A%84%E5%89%8D%E4%B8%96%E4%BB%8A%E7%94%9F)
  - [二十世纪五十年代:推理期](#%E4%BA%8C%E5%8D%81%E4%B8%96%E7%BA%AA%E4%BA%94%E5%8D%81%E5%B9%B4%E4%BB%A3%E6%8E%A8%E7%90%86%E6%9C%9F)
  - [二十世纪七十年代中期:知识期](#%E4%BA%8C%E5%8D%81%E4%B8%96%E7%BA%AA%E4%B8%83%E5%8D%81%E5%B9%B4%E4%BB%A3%E4%B8%AD%E6%9C%9F%E7%9F%A5%E8%AF%86%E6%9C%9F)
  - [二十世纪八十年代:从样例中学习](#%E4%BA%8C%E5%8D%81%E4%B8%96%E7%BA%AA%E5%85%AB%E5%8D%81%E5%B9%B4%E4%BB%A3%E4%BB%8E%E6%A0%B7%E4%BE%8B%E4%B8%AD%E5%AD%A6%E4%B9%A0)
    - [符号主义学习](#%E7%AC%A6%E5%8F%B7%E4%B8%BB%E4%B9%89%E5%AD%A6%E4%B9%A0)
    - [连接主义学习](#%E8%BF%9E%E6%8E%A5%E4%B8%BB%E4%B9%89%E5%AD%A6%E4%B9%A0)
  - [二十世纪九十年代中期:统计学习](#%E4%BA%8C%E5%8D%81%E4%B8%96%E7%BA%AA%E4%B9%9D%E5%8D%81%E5%B9%B4%E4%BB%A3%E4%B8%AD%E6%9C%9F%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0)
  - [二十一世纪:深度学习](#%E4%BA%8C%E5%8D%81%E4%B8%80%E4%B8%96%E7%BA%AA%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0)
- [Regression,Neural Network & Deep Learning:回归，神经网络与机器学习](#regressionneural-network-&-deep-learning%E5%9B%9E%E5%BD%92%EF%BC%8C%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0)
  - [Regression:无意义的等式](#regression%E6%97%A0%E6%84%8F%E4%B9%89%E7%9A%84%E7%AD%89%E5%BC%8F)
  - [Neural Network:连接主义](#neural-network%E8%BF%9E%E6%8E%A5%E4%B8%BB%E4%B9%89)
  - [Deep Learning:深层连接](#deep-learning%E6%B7%B1%E5%B1%82%E8%BF%9E%E6%8E%A5)
- [Machine Learning Algorithms:常见的机器学习算法分类](#machine-learning-algorithms%E5%B8%B8%E8%A7%81%E7%9A%84%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%AE%97%E6%B3%95%E5%88%86%E7%B1%BB)
  - [Grouped By Learning Style:根据学习风格分类](#grouped-by-learning-style%E6%A0%B9%E6%8D%AE%E5%AD%A6%E4%B9%A0%E9%A3%8E%E6%A0%BC%E5%88%86%E7%B1%BB)
    - [Supervised Learning:有监督学习](#supervised-learning%E6%9C%89%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0)
    - [Semi-Supervised Learning:半监督学习](#semi-supervised-learning%E5%8D%8A%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0)
    - [Unsupervised Learning:无监督学习](#unsupervised-learning%E6%97%A0%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0)
  - [Grouped By Similarity:根据算法相似度分类](#grouped-by-similarity%E6%A0%B9%E6%8D%AE%E7%AE%97%E6%B3%95%E7%9B%B8%E4%BC%BC%E5%BA%A6%E5%88%86%E7%B1%BB)
    - [Regression Algorithms:回归拟合](#regression-algorithms%E5%9B%9E%E5%BD%92%E6%8B%9F%E5%90%88)
    - [Instance-based Algorithms:基于实例的算法](#instance-based-algorithms%E5%9F%BA%E4%BA%8E%E5%AE%9E%E4%BE%8B%E7%9A%84%E7%AE%97%E6%B3%95)
    - [Decision Tree Algorithms:决策树算法](#decision-tree-algorithms%E5%86%B3%E7%AD%96%E6%A0%91%E7%AE%97%E6%B3%95)
    - [Bayesian Algorithms:贝叶斯算法](#bayesian-algorithms%E8%B4%9D%E5%8F%B6%E6%96%AF%E7%AE%97%E6%B3%95)
    - [Clustering Algorithms:聚类算法](#clustering-algorithms%E8%81%9A%E7%B1%BB%E7%AE%97%E6%B3%95)
    - [Artificial Neural Network Algorithms:人工神经网络算法](#artificial-neural-network-algorithms%E4%BA%BA%E5%B7%A5%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%AE%97%E6%B3%95)
    - [Deep Learning Algorithms:深度学习算法](#deep-learning-algorithms%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%AE%97%E6%B3%95)
    - [Support Vector Machines](#support-vector-machines)
    - [Ensemble Algorithms](#ensemble-algorithms)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

> 本文会随着笔者自己认知的变化而不断更新，有兴趣的话可以关注笔者的专栏或者Github。

# Introduction

互联网的迅猛发展催生了数据的爆炸式增长。面对海量数据，如何挖掘数据的架子，成为一个越来越重要的问题。首先，对于数据挖掘的概念，目前比较广泛认可的一种解释如下：

> Data Mining is the use of efficient techniques for the analysis of very large collections of data and the extraction of useful and possibly unexpected patterns  in data。



数据挖掘是一种通过分析海量数据，从数据中提取潜在的但是非常有用的模式的技术。数据挖掘的任务可以分为预测性任务和描述性任务，预测性任务主要是预测可能出现的情况；描述性任务则是发现一些人类可以解释的模式或者规律。数据挖掘中比较常见的任务包括分类、聚类、关联规则挖掘、时间序列挖掘、回归等，其中分类、回归属于预测性任务，聚类、关联规则挖掘、时间序列分析等则都是解释性任务。而什么又是机器学习呢？笔者在这里引用[有趣的机器学习概念纵览：从多元拟合，神经网络到深度学习，给每个感兴趣的人](https://segmentfault.com/a/1190000005746236)中的定义：

> Machine learning is the idea that there are generic algorithms that can tell you something interesting about a set of data without you having to write any custom code specific to the problem. Instead of writing code, you feed data to the generic algorithm and it builds its own logic based on the data.



Machine Learning即是指能够帮你从数据中寻找到感兴趣的部分而不需要编写特定的问题解决方案的通用算法的集合。通用的算法可以根据你不同的输入数据来自动地构建面向数据集合最优的处理逻辑。举例而言，算法中一个大的分类即分类算法，它可以将数据分类到不同的组合中。而可以用来识别手写数字的算法自然也能用来识别垃圾邮件，只不过对于数据特征的提取方法不同。相同的算法输入不同的数据就能够用来处理不同的分类逻辑。换一个形象点的阐述方式，对于某给定的任务T，在合理的性能度量方案P的前提下，某计算机程序可以自主学习任务T的经验E；随着提供合适、优质、大量的经验E，该程序对于任务T的性能逐步提高。即随着任务的不断执行，经验的累积会带来计算机性能的提升。

在二十世纪九十年代中期之后，统计学习渐渐成为机器学习的主流方向，具体关于机器学习的发展历程可以参见下文的机器学习的前世今生章节。

![](http://7xiegq.com1.z0.glb.clouddn.com/1-YXiclXZdJQVJZ0tQHCv5zw.png)



论及数据挖掘与机器学习的关系，笔者更倾向于使用数据科学这个概念来进行一个总括，在笔者的归纳中，数据科学泛指一切可以从数据中获取信息的技术与研究，不过注意和Infrastructure部分进行区分，笔者的数据科学部分知识体系，会包含以下部分：

- Methodology:方法论

    - DataProcess:数据预处理

    - MachineLearning:机器学习

    - NLP:自然语言处理

    - Statistics:数理统计

    - DeepLearning:深度学习

- Application:应用

    - Classification:分类

    -  CommunityDetection:社团发现

    - IntelligentAssistant:智能辅助

    - Personas:用户画像

    -  Recognition:模式识别

    - RecommendSystem:推荐系统

- CrawlerSE:爬虫与搜索引擎

- DataVisualization:数据可视化

- Toolkits:工具集



综上所述，数据科学涵括数据挖掘，机器学习是数据挖掘的重要手段之一，而统计学习是目前机器学习的主流方向。





## Reference



### Tutorials & Docs

- [有趣的机器学习概念纵览：从多元拟合，神经网络到深度学习，给每个感兴趣的人](https://segmentfault.com/a/1190000005746236)

- [A Few Useful Things to Know about Machine Learning By Pedro Domingos](http://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf)

- [机器学习温和指南](http://www.csdn.net/article/2015-09-08/2825647#rd?sukey=b0cb5c5b9e50130317c130ba2243b8e573404f264fe1467cde82d02d2529978108ee91bb9a2bbc0d81118e2db77390ca)
- [机器学习与数据挖掘的学习路线图](http://mp.weixin.qq.com/s?__biz=MzA3MDg0MjgxNQ==&mid=2652389718&idx=1&sn=9f0c3a6f525f4c28a504e7475efc8f02&scene=0#wechat_redirect)



#### Video Courses:视频教程

- [Stanford-UFLDL教程](http://ufldl.stanford.edu/wiki/index.php/UFLDL%E6%95%99%E7%A8%8B)
- [机器学习视频教程——小象训练营，这里面提供了一系列的课程的说明](http://www.chinahadoop.cn/course/423)

### Practices & Resources



- [DataScience/MachineLearning Toolkits Index:笔者总结的数据科学/机器学习中常用工具集合/示例索引]()

- [awesome-datascience](https://github.com/okulbilisim/awesome-datascience)

- [data-science-ipython-notebooks](https://github.com/donnemartin/data-science-ipython-notebooks)


#### Collection

- [部分免费的DataScience方面的书籍](http://www.learndatasci.com/free-books/)

- [dive-into-machine-learning](https://github.com/hangtwenty/dive-into-machine-learning)

- [a-tour-of-machine-learning-algorithms](http://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/)

- [史上最全的机器学习资料](https://yq.aliyun.com/articles/43089)



### Blogs & News

- [Data science blogs](https://github.com/rushter/data-science-blogs)

- [zouxy09的专栏](http://blog.csdn.net/zouxy09)
- [一个机器学习算法的博客](http://www.algorithmdog.com/)


### Books & Tools

- [2016-周志华-机器学习]()

- [2012-李航-统计学习方法](https://coding.net/u/hoteam/p/Cache/git/raw/master/2016/6/4/2012%25E6%259D%258E%25E8%2588%25AA%25E7%25BB%259F%25E8%25AE%25A1%25E5%25AD%25A6%25E4%25B9%25A0%25E6%2596%25B9%25E6%25B3%2595.pdf)

- [2011-范明译本-数据挖掘导论完整版](http://o6v08w541.bkt.clouddn.com/%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98%E5%AF%BC%E8%AE%BA%E7%BE%8EPNTan%E8%8C%83%E6%98%8E%E8%AF%912011.pdf)

- [2012-吴军-数学之美](https://coding.net/u/hoteam/p/Cache/git/raw/master/2016/6/4/%25E6%2595%25B0%25E5%25AD%25A6%25E4%25B9%258B%25E7%25BE%258E.pdf)

- [2011-Jiawei Han-Data Mining Concepts and Techniques](https://coding.net/u/hoteam/p/Cache/git/raw/master/2016/6/4/DataMiningConceptsandTechniques3e2011.pdf)

- [2011-DataMining Practical Machine Learning Tools and Techniques](https://coding.net/u/hoteam/p/Cache/git/raw/master/2016/6/4/DataMiningPracticalMachineLearningToolsandTechniques3e2011.pdf)

- [2010-Ethem Alpaydin-Introduction To Machine Learning Second Edition](https://coding.net/u/hoteam/p/Cache/git/raw/master/2016/6/4/IntroductiontoMachineLearning2e2010.pdf)

- [2010-Jure-Mining of Massive DataSets](https://coding.net/u/hoteam/p/Cache/git/raw/master/2016/6/4/MiningofMassiveDatasets.pdf)

- [2008-The Elements of Statistical Learning Data Mining Inference and Prediction](https://coding.net/u/hoteam/p/Cache/git/raw/master/2016/6/4/TheElementsofStatisticalLearningDataMiningInferenceandPrediction.pdf)

- [2001-Principles of DataMining DJHand 2001](https://coding.net/u/hoteam/p/Cache/git/raw/master/2016/6/4/PrinciplesofDataMiningDJHand2001.pdf)

- [Introduction to Machine Learning with Python](https://github.com/amueller/introduction_to_ml_with_python)

- [A Programmer's Guide to Data Mining](http://guidetodatamining.com/):一本开源的介绍了数据挖掘方面的书，包括了推荐系统、协同过滤、分类、Native Bayes、基本的NLP以及聚类

- [Standford-The Elements of Statistical Learning Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/local.ftp/Springer/OLD/ESLII_print4.pdf)


# DataScience LifeCycle:一次数据研究的生命周期

> [DataScience-Life-Cycle](https://github.com/okulbilisim/awesome-datascience/blob/master/DataScience-Life-Cycle.md)

DataScience是从数据中获取价值的最重要的流行方法之一。


## Identify Problem:定位问题

这个环节最重要的是需求和数据的匹配。首先需要明确需求，比如分类问题中需要判断是否有带标注的训练数据集，否则，无法按照有监督的分类算法来解决。此外，数据的规模、重要Feature的覆盖率等等，也都是要考虑的问题。



## Data PreProcess:数据预处理

###  数据集成、数据冗余与数值冲突

数据挖掘中准备数据的时候，需要尽可能的将相关数据集成在一起。如果集成的数据中，有两列或者多列值一样，则不可避免地会产生数值冲突或者数据冗余，可能需要根据数据的质量来决定保留冲突中的哪一列。

### 数据采样

一般来说，有效的采样方式如下：如果样本是有代表性的，则使用样本数据和使用整个数据集的效果几乎是一样的。抽样方法有很多，首先考虑的是有放回的采样，以及具体选择哪种采样方式。

### 数据清洗、缺失值处理与噪声数据

现实世界中的数据，是真实的数据，不可避免地会存在各种各样异常的情况。比如某一列的值缺失或者某列的值是异常的。所以，我们需要在数据预处理阶段进行数据清洗，来减少噪音数据对于模型训练和预测结果的影响。

## Feature Engineering:特征工程

数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限。

> feature engineering is another topic which doesn’t seem to merit any review papers or books, or even chapters in books, but it is absolutely vital to ML success. […] Much of the success of machine learning is actually success in engineering features that a learner can understand.



特征是对于所需解决问题有用的属性。比如在计算机视觉领域，图片作为研究对象，可能图片中的一个线条就是一个特征；在自然语言处理领域，研究对象是文档，文档中一个词语出现的次数就是一个特征；在语音识别领域，研究对象是一段话，那么一个音频位就是一个特征。

### 特征的提取、选择和构造

既然特征是对我们所解决的问题最有用的属性，首先我们需要处理的是根据原始数据抽取出所需要的特征。不过需要注意的是并不是所有的特征对所解决的问题影响一样大，有些特征可能对问题产生特别大的影响，但是有些则影响甚微，和所解决的问题不相关的特征需要被剔除掉。因此，我们需要针对所解决的问题选择最有用的特征集合，一般可以通过相关系数等方式来计算特征的重要性。

当然，有些模型本身会输出feature重要性，譬如Random Forest算法，而对于图片、音频等原始数据形态特别大的对象，则可能需要采用像PCA这样的自动降维技术。另外，还可能需要本人对数据和所需解决的问题有深入的理解，能够通过特征组合等方式构造出新的特征，这也正是特征工程被称为一门艺术的原因。

## Model Selection&Evaluation:模型选择与评估

一般来说，对于算法与模型选型的考虑，可能有：

- 训练集的大小

- 特征的维度大小

- 所解决问题是否是线性可分的

- 所有的特征是独立的吗

- 是否需要考虑过拟合

- 性能要求



在实际的工作中，我们选择算法与模型时候会考虑到奥卡姆剃刀原理：

> Occam's Razor principle:use the least complicated algorithm that can address your needs and only go for something more complicated if strictly necessary



业界比较通用的算法选择一般是这样的规律：如果LR可以，则用LR；如果LR不合适，则选择Ensemble方式；如果Ensemble方式不合适，则考虑是否尝试Deep Learning。



# History of Machine Learning:机器学习的前世今生

> 本部分参考了周志华教授的著作：《机器学习》

- [百度百家：盘点机器学习领域的五大流派](http://valser.org/thread-757-1-1.html)



机器学习是人工智能研究发展到一定阶段的必然产物，本章仅从笔者的视角对机器学习这五十年来的发展进行一个略述，疏漏错误之处烦请指正。下面这幅漫画中就展示了一个无奈的问题，三岁幼童可以轻松解决的问题却需要最顶尖的科学家花费数十年的光阴，或许机器学习离我们在电影里看到的那样还有很长一段路要走。

![](http://7xi5sw.com1.z0.glb.clouddn.com/1-wUZiI2Mg2cncuMWWXIiBgQ.png)

知识来源于哪里？知识来源于进化、经验、文化和计算机。对于知识和计算机的关系，可以引用Facebook人工智能实验室负责人Yann LeCun的一段话：将来，世界上的大部分知识将由机器提取出来，并且将长驻与机器中。而帮助计算机获取新知识，可以通过以下五种方法来实现：

- 填充现存知识的空白

- 对大脑进行仿真

- 对进化进行模拟

- 系统性的减少不确定性

- 注意新旧知识之间的相似点



对应以上这几种知识获取的途径，我们可以认为常见的人工智能的方向有：

| 派别                   | 起源     | 擅长算法                          |

| -------------------- | ------ | ----------------------------- |

| 符号主义（Symbolists）     | 逻辑学、哲学 | 逆演绎算法（Inverse deduction）      |

| 联结主义（Connectionists） | 神经科学   | 反向传播算法（Backpropagation）       |

| 进化主义（Evolutionaries） | 进化生物学  | 基因编程（Genetic programming）     |

| 贝叶斯派（Bayesians）      | 统计学    | 概率推理（Probabilistic inference） |

| Analogizer           | 心理学    | 核机器（Kernel machines）          |



## 二十世纪五十年代:推理期

二十世纪五十年代到七十年代初，人工智能研究处于”推理期“，彼时人们以为只要能赋予机器逻辑推理的能力，机器就能具有智能。这一阶段的代表性作品有A. Newell和H. Simon的“逻辑理论家”程序，该程序于1952年证明了罗素和怀特海的名著《数学原理》中的38条定理，在1963年证明了全部的52条定理。不过随着时间的发展，人们渐渐发现仅具有逻辑推理能力是远远实现不了人工智能的。



## 二十世纪七十年代中期:知识期

从二十世纪七十年代中期开始，人工智能研究进入了“知识期”，在这一时期，大量的专家系统问世，在很多应用领域取得了大量成果。在本阶段诞生的技术的一个鲜明的代表就是模式识别，它强调的是如何让一个计算机程序去做一些看起来很“智能”的事情，例如识别“3”这个数字。而且在融入了很多的智慧和直觉后，人们也的确构建了这样的一个程序。从这个时代诞生的模式识别领域最著名的书之一是由Duda & Hart执笔的“模式识别（Pattern Classification）”。 对基础的研究者来说，仍然是一本不错的入门教材。不过对于里面的一些词汇就不要太纠结了，因为这本书已经有一定的年代了，词汇会有点过时。自定义规则、自定义决策，以及自定义“智能”程序在这个任务上，曾经都风靡一时。有趣的是笔者在下文中也会介绍如何用深度学习网络去识别手写的数字，有兴趣的朋友可以去探究下使用模式识别与深度学习相比，同样是识别手写数字上的差异。

不过，专家系统面临“知识工程瓶颈”，即由人来把知识总结出来再教给计算机是相当困难的，于是人们开始考虑如果机器能够自己学习知识，该是一件多么美妙的事。



## 二十世纪八十年代:从样例中学习

R.S.Michalski等人将机器学习分为了“从样例中学习”、“在问题求解和规划中学习”、“通过观察和发现学习”、“从指令中学习”等类别；E.A.Feigenbaum等人在著作《人工智能手册》中，则把机器学习划分为了“机械学习”、“示教学习”、“类比学习”和“归纳学习”。机械学习又被称为死记硬背式学习，即把外界输入的信息全部记录下来，在需要时原封不动地取出来使用，这实际上没有进行真正的学习，仅仅是在进行信息存储和检索；示教学习和类比学习类似于R.S.Michalski等人所说的从指令中学习和通过观察和发现学习。归纳学习则相当于从样例中学习，即从训练样本中归纳出学习结果。二十世纪八十年代以来，被研究最多、应用最广的是“从样例中学习”，也就是广泛的归纳学习，它涵盖了监督学习、无监督学习等。



### 符号主义学习

在二十世纪八十年代，从样例中学习的一大主流就是符号主义学习，其代表包括决策树和基于逻辑的学习。符号学习一个直观的流程可以参考下图：

![](http://f.hiphotos.baidu.com/news/w%3D638/sign=0cd9875ed6a20cf44690fddc4e084b0c/cf1b9d16fdfaaf51a68581668a5494eef11f7a80.jpg)

典型的决策树学习以信息论为基础，以信息熵的最小化为目标，直接模拟了人类对概念进行判定的树形流程。基于逻辑的学习的著名代表是归纳逻辑程序设计Inductive Logic Programming，简称ILP，可以看做机器学习与逻辑程序设计的交叉。它使用一阶逻辑，即谓词逻辑来进行知识表示，通过修改和扩充逻辑表达式来完成对于数据的归纳。符号主义学习占据主流地位与前几十年人工智能经历的推理期和知识期密切相关，最后，可以来认识几位符号主义的代表人物：

![](http://g.hiphotos.baidu.com/news/w%3D638/sign=1732fe15c5cec3fd8b3ea476ee89d4b6/0e2442a7d933c8957f639ba2d71373f0830200cd.jpg)



### 连接主义学习

二十世纪九十年代中期之前，从样例中学习的另一主流技术是基于神经网络的连接主义学习。下图就是典型的神经元、神经网络与著名的BP算法的示例。

![](http://a.hiphotos.baidu.com/news/w%3D638/sign=1dfae1c32e34349b74066d86f1e81521/d1a20cf431adcbef4ede7edfaaaf2edda2cc9f04.jpg)

![](http://g.hiphotos.baidu.com/news/w%3D638/sign=f96a1ed11930e924cfa49f3274096e66/42166d224f4a20a4b8413f1e96529822730ed088.jpg)

![](http://g.hiphotos.baidu.com/news/w%3D638/sign=8c3defdace95d143da76e7204bf18296/a8773912b31bb0513683b0f1307adab44bede040.jpg)

与符号主义学习能产生明确的概念表示不同，连接主义学习产生的是黑箱模型，因此从知识获取的角度来看，连接主义学习技术有明显弱点。然而，BP一直是被应用的最广泛的机器学习算法之一，在很多现实问题上发挥作用。连接主义学习的最大局限是其试错性。简单来说，其学习过程设计大量的参数，而参数的设置缺乏理论指导，主要靠手工调参；夸张一点来说，参数调节上失之毫厘，学习结果可能谬以千里。



## 二十世纪九十年代中期:统计学习

二十世纪九十年代中期，统计学习闪亮登场并且迅速占据主流舞台，代表性技术是支持向量机(Support Vector Machine)以及更一般的核方法(Kernel Methods)。正是由于连接主义学习技术的局限性凸显，人们才把目光转向以统计学习理论为直接支撑的统计学习技术。

![](http://h.hiphotos.baidu.com/news/w%3D638/sign=46f5107e5fee3d6d22c684c87b146d41/6159252dd42a2834235335d25db5c9ea14cebf75.jpg)



## 二十一世纪:深度学习

> 深度学习掀起的热潮也许大过它本身真正的贡献，在理论和技术上并没有太多的创新，只不过是由于硬件技术的革命，计算机的速度大大提高了，使得人们有可能采用原来复杂度很高的算法，从而得到比过去更精细的结果。



二十一世纪初，连接主义学习又卷土重来，掀起了以深度学习为名的热潮。所谓深度学习，狭义的说就是“很多层”的神经网络。在若干测试和竞赛上，尤其是涉及语音、图像等复杂对象的应用中，深度学习技术取得了优越性能。之前的机器学习技术在应用中要取得好的性能，对于使用者的要求较高。而深度学习技术涉及的模型复杂度非常高，以至于只要下功夫“调参”，把参数调节好，性能往往就好。深度学习虽然缺乏严格的理论基础，但是显著降低了机器学习应用者的门槛，为机器学习走向工程实践带来了便利。深度学习火热的原因有：

- 数据大了，计算能力抢了，深度学习模型拥有大量参数，若数据样本少，则很容易过拟合。

- 由于人类进入了大数据时代，数据储量与计算设备都有了大发展，才使得连接主义学习技术焕发了又一春。



笔者作为比较纯粹的偏工程的研究汪，并没有真正大规模使用过深度学习集群，但是，譬如Tensorflow，它的编程易用性和同等情况下的效果还是会大于比对的算法。关于上面一段话，跪求大神打脸指正。(￣ε(#￣)☆╰╮(￣▽￣///)





# Regression,Neural Network & Deep Learning:回归，神经网络与机器学习

> 本部分节选自笔者另一篇文章：[有趣的机器学习概念纵览：从多元拟合，神经网络到深度学习，给每个感兴趣的人](https://segmentfault.com/a/1190000005746236)



## Regression:无意义的等式

首先我们来看一个真实的例子，假设你是一位成功的房地产中介，你的事业正在蒸蒸日上，现在打算雇佣更多的中介来帮你一起工作。不过问题来了，你可以一眼看出某个房子到底估值集合，而你的实习生们可没你这个本事。为了帮你的实习生尽快适应这份工作，你打算写个小的APP来帮他们根据房子的尺寸、邻居以及之前卖的类似的屋子的价格来评估这个屋子值多少钱。因此你翻阅了之前的资料，总结成了下表：

![](http://7xiegq.com1.z0.glb.clouddn.com/1-ZWYX9nwsDFaNOW4jOrHDkQ.png)

利用这些数据，我们希望最后的程序能够帮我们自动预测一个新的屋子大概能卖到多少钱：

![](http://7xiegq.com1.z0.glb.clouddn.com/1-V0OXzLOPtpU13MVVrlZJjA.png)

从知识学习与人类思维的角度来说，我们会根据房屋的几个特性来推导出最后的房屋价格，根据经验我们可能得到如下等式：

```

def estimate_house_sales_price(num_of_bedrooms, sqft, neighborhood):

  price = 0

  # 俺们这嘎达，房子基本上每平方200

  price_per_sqft = 200

  if neighborhood == "hipsterton":

    # 市中心会贵一点

    price_per_sqft = 400

  elif neighborhood == "skid row":

    # 郊区便宜点

    price_per_sqft = 100

  # 可以根据单价*房子大小得出一个基本价格

  price = price_per_sqft * sqft

  # 基于房间数做点调整

  if num_of_bedrooms == 0:

    # 没房间的便宜点

    price = price — 20000

  else:

    # 房间越多一般越值钱

    price = price + (num_of_bedrooms * 1000)

 return price

```

这就是典型的简答的基于经验的条件式判断，你也能通过这种方法得出一个较好地模型。不过如果数据多了或者价格发生较大波动的时候，你就有心无力了。而应用机器学习算法则是让计算机去帮你总结出这个规律，大概如下所示：

```

def estimate_house_sales_price(num_of_bedrooms, sqft, neighborhood):

  price = <computer, plz do some math for me>

  return price

```

通俗的理解，价格好比一锅炖汤，而卧室的数量、客厅面积以及邻近的街区就是食材，计算机帮你自动地根据不同的食材炖出不同的汤来。如果你是喜欢数学的，那就好比有三个自变量的方程，代码表述的话大概是下面这个样子：

```

def estimate_house_sales_price(num_of_bedrooms, sqft, neighborhood):

 price = 0

 # a little pinch of this

 price += num_of_bedrooms * .841231951398213

 # and a big pinch of that

 price += sqft * 1231.1231231

 # maybe a handful of this

 price += neighborhood * 2.3242341421

 # and finally, just a little extra salt for good measure

 price += 201.23432095

 return price

```

这里不做深入讲解，只是观察上述代码可以发现，在大量数据的情况下，最终可以形成一个带权重和的等式，而其中的参数，从直观意义上来讲，可能毫无意义。



## Neural Network:连接主义

上述算法只能处理一些较为简单的问题，即结果与输入的变量之间存在着某些线性关系。Too young,Too Simple,真实的房价和这些可不仅仅只有简单的线性关系，譬如邻近的街区这个因子可能对面积较大和面积特别小的房子有影响，但是对于那些中等大小的毫无关系，换言之，price与neighborhood之间并不是线性关联，而是类似于二次函数或者抛物线函数图之间的非线性关联。这种情况下，我们可能得到不同的权重值（形象来理解，可能部分权重值是收敛到某个局部最优）：

![](http://7u2q25.com1.z0.glb.clouddn.com/1-hOemQF_v42KHlMyiqcQNyQ.png)

现在等于每次预测我们有了四个独立的预测值，下一步就是需要将四个值合并为一个最终的输出值：

![](http://7u2q25.com1.z0.glb.clouddn.com/1-VeS0ziSjogCQThPYZh0TIQ.png)

我们将上文提到的两个步骤合并起来，大概如下图所示：

![](http://7u2q25.com1.z0.glb.clouddn.com/1-Lt8RZaeQ6f6B_eA1oD32JQ.png)

咳咳，没错，这就是一个典型的神经网络，每个节点接收一系列的输入，为每个输入分配权重，然后计算输出值。通过连接这一系列的节点，我们就能够为复杂的函数建模。



## Deep Learning:深层连接

我们在文首就强调过，机器学习是通用算法加上特定的数据最终形成一套特定的模型，那么我们将上面房产的数据修正为手写数字，大概就像如下这样：

![](http://7xi5sw.com1.z0.glb.clouddn.com/1-jYKYXkfI4iaE6qg-dEUEcQ.jpeg)

在这里我们希望用神经网络来处理图片，第二步就是需要将一张图片转化为数字的组合，即是计算机可以处理的样子。表担心，这一步还是很简单的。对于电脑而言，一张图片就是一个多维的整型数组，每个元素代表了每个像素的模糊度，大概是这样子：

![](http://7xi5sw.com1.z0.glb.clouddn.com/1-zY1qFB9aFfZz66YxxoI2aw.gif)

为了能够将图片应用到我们的神经网络模型中，我们需要将1818像素的图片转化为324个数字：

![](http://7xiegq.com1.z0.glb.clouddn.com/1-UDgDe_-GMs4QQbT8UopoGA.png)

这次的共有324个输入，我们需要将神经网络扩大化转化为324个输入的模型：

![](http://7xiegq.com1.z0.glb.clouddn.com/1-b31hqXiBUjIXo2HSn_grFw.png)

用这种方法，我们可以方便地创建无限的训练数据。数据有了，我们也需要来扩展下我们的神经网络，从而使它能够学习些更复杂的模式。具体而言，我们需要添加更多的中间层：

![](http://7xiegq.com1.z0.glb.clouddn.com/1-wfmpsoFqWKC7VadjTJxwnQ.png)

这个呢，就是我们所谓的深度神经网络，因为它比传统的神经网络有更多的中间层。



# Machine Learning Algorithms:常见的机器学习算法分类

- [a-tour-of-machine-learning-algorithms](http://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/)

算法基本上从功能或者形式上来分类。比如，基于树的算法，神经网络算法。这是一个很有用的分类方式，但并不完美。因为有许多算法可以轻易地被分到两类中去，比如说Learning Vector Quantization就同时是神经网络类的算法和基于实例的方法。正如机器学习算法本身没有完美的模型一样，算法的分类方法也没有完美的。

## Grouped By Learning Style:根据学习风格分类

### Supervised Learning:有监督学习

输入数据被称为训练数据，并且有已知的结果或被标记。比如说一封邮件是否是垃圾邮件，或者说一段时间内的股价。模型做出预测，如果错了就会被修正，这个过程一直持续到对于训练数据它能够达到一定的正确标准。问题例子包括分类和回归问题，算法例子包括逻辑回归和反向神经网络。

![](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Supervised-Learning-Algorithms.png)

有监督学习的通俗例子可以以人类认知月亮的过程为例，小时候会认为黑夜里天空中明亮的圆盘状物体为月亮，后来随着时间的推移，发现圆盘状这个特点并非一个决定性因素。



### Semi-Supervised Learning:半监督学习

![](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Semi-supervised-Learning-Algorithms.png)

输入数据是被标记的和不被标记的数据的混合，有一些预测问题但是模型也必须学习数据的结构和组成。问题例子包括分类和回归问题，算法例子基本上是无监督学习算法的延伸。



### [Unsupervised Learning](https://github.com/okulbilisim/awesome-datascience/blob/master/Algorithms.md#unsupervised-learning):无监督学习

输入数据没有被标记，也没有确定的结果。模型对数据的结构和数值进行归纳。问题例子包括Association rule learning和聚类问题，算法例子包括 Apriori 算法和K-均值算法。

![](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Unsupervised-Learning-Algorithms.png)



## Grouped By Similarity:根据算法相似度分类

### Regression Algorithms:回归拟合
![Regression Algorithms](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Regression-Algorithms.png)


回归主要用于对变量之间的关系进行建模，并且使用某个损失函数或者代价函数作为某个变量组合的评估标准。回归方法是统计的一个重要组成部分，目前主要用于统计机器学习中。有时候这个概念会混乱，因为有时候会用回归去描述一类问题，有时候也会用回归去描述一类算法，而实际上回归更多的指的是一个处理过程。常见的回归算法有:

- Ordinary Least Squares Regression (OLSR)
- Linear Regression
- Logistic Regression
- Stepwise Regression
- Multivariate Adaptive Regression Splines (MARS)
- Locally Estimated Scatterplot Smoothing (LOESS)

### Instance-based Algorithms:基于实例的算法
![Instance-based Algorithms](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Instance-based-Algorithms.png)



基于实例的学习模型主要是根据训练实例或者样本的特性进行模型构建，因此对于此类算法训练数据至关重要。此类型算法通常会对测试数据建立一个统一的数据库，然后通过比较新老数据之间的相似度来进行匹配预测。鉴于这个特性，基于实例的算法又被成为“赢者通吃”或者“基于记忆”的学习。此类算法的核心在于存储实例的表示方法与实例间相似度的度量方法，最常见的基于实例的学习算法有:

- k-Nearest Neighbour (kNN)
- Learning Vector Quantization (LVQ)
- Self-Organizing Map (SOM)
- Locally Weighted Learning (LWL)

### Decision Tree Algorithms:决策树算法

![Decision Tree Algorithms](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Decision-Tree-Algorithms.png)



决策树即利用数据中属性的真实值来构建一个决策模型，决策树算法也主要用于分类与回归问题。决策树算法一般速度和准确度都不错，因此在机器学习中很受欢迎。常见的机器学习算法有：

- Classification and Regression Tree (CART)
- Iterative Dichotomiser 3 (ID3)
- C4.5 and C5.0 (different versions of a powerful approach)
- Chi-squared Automatic Interaction Detection (CHAID)
- Decision Stump
- M5
- Conditional Decision Trees

### Bayesian Algorithms:贝叶斯算法

![Bayesian Algorithms](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Bayesian-Algorithms.png)

贝叶斯算法即是基于贝叶斯公式及其变种而用于常见的分类与聚类问题的算法。常见的贝叶斯算法有：


- Naive Bayes
- Gaussian Naive Bayes
- Multinomial Naive Bayes
- Averaged One-Dependence Estimators (AODE)
- Bayesian Belief Network (BBN)
- Bayesian Network (BN)

### Clustering Algorithms:聚类算法

![Clustering Algorithms](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Clustering-Algorithms.png)

聚类，与回归一样同时描述了一系列问题与一系列算法的集合。聚类算法通常地基于中心与层次化进行建模，基本上所有的方法都使用数据的内部结构来将数据组合进入不同的社群中，常见的聚类算法有：


- k-Means
- k-Medians
- Expectation Maximisation (EM)
- Hierarchical Clustering

### Artificial Neural Network Algorithms:人工神经网络算法

![Artificial Neural Network Algorithms](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Artificial-Neural-Network-Algorithms.png)
人工神经网络算法主要受生物学中的人工神经网络的启发，需要注意，本文将深度学习剥离为一个单独的部分进行描述。常见的人工神经网络算法有：

- Perceptron
- Back-Propagation
- Hopfield Network
- Radial Basis Function Network (RBFN)

### Deep Learning Algorithms:深度学习算法

![Deep Learning Algorithms](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Deep-Learning-Algorithms.png)

深度学习算法是对于人工神经网络算法的一个扩展，基于现代大量廉价的计算能力的组合从而实现对于复杂模型的求解。深度学习算法主要是基于构建大型、复杂的神经网络，很多方法都是为了处理只含有少量标签数据的海量数据集的半监督学习方法。常见的深度学习算法有：

- Deep Boltzmann Machine (DBM)
- Deep Belief Networks (DBN)
- Convolutional Neural Network (CNN)
- Stacked Auto-Encoders

### Support Vector Machines

### Ensemble Algorithms

![Ensemble Algorithms](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Ensemble-Algorithms.png)

组合方式主要分为Bagging和Boosting。Bagging是Bootstrapped Aggregation的缩写，其基本原理是让学习算法训练多轮，每轮的训练集由从初始的训练集中随机取出的N个训练样本组成（有放回的随机抽样），训练之后可得到一个预测函数集合，通过投票的方式决定预测结果。而Boosting中主要的是AdaBoost(Adaptive Boosting)。基本原理是初始化时对每一个训练样本赋相等的权重$1/n$，然后用学习算法训练多轮，每轮结束后，对训练失败的训练样本赋以较大的权重，也就是让学习算法在后续的学习中集中对比较难的训练样本进行学习，从而得到一个预测函数的集合。每个预测函数都有一定的权重，预测效果好的函数权重较大，反之较小，最终通过有权重的投票方式来决定预测结果。而Bagging与Boosting的主要区别在于：

- 取样方式不同。Bagging采用均匀取样，而Boosting根据错误率来取样，因此理论上来说Boosting的分类精度要优于Bagging。

- 训练集的选择方式不同。Bagging的训练集的选择是随机的，各轮训练集之间相互独立，而Boosting的各轮训练集的选择与前面的学习结果有关。

- 预测函数不同。Bagging的各个预测函数没有权重，而Boosting是有权重的。Bagging的各个预测函数可以并行生成，而Boosting的各个预测函数只能顺序生成。



总结而言，对于像神级网络这样及其耗时的学习方法，Bagging可以通过并行训练节省大量时间的开销。Bagging和Boosting都可以有效地提高分类的准确性。在大多数数据集合中，Boosting的准确性比Bagging要高。

- Boosting
- Bootstrapped Aggregation (Bagging)
- AdaBoost
- Stacked Generalization (blending)
- Gradient Boosting Machines (GBM)
- Gradient Boosted Regression Trees (GBRT)
- Random Forest



















