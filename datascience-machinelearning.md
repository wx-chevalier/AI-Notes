# Introduction

互联网的迅猛发展催生了数据的爆炸式增长。面对海量数据，如何挖掘数据的架子，成为一个越来越重要的问题。首先，对于数据挖掘的概念，目前比较广泛认可的一种解释如下：Data Mining is the use of efficient techniques for the analysis of very large collections of data and the extraction of useful and possibly unexpected patterns  in data。数据挖掘是一种通过分析海量数据，从数据中提取潜在的但是非常有用的模式的技术。数据挖掘的任务可以分为预测性任务和描述性任务，预测性任务主要是预测可能出现的情况；描述性任务则是发现一些人类可以解释的模式或者规律。数据挖掘中比较常见的任务包括分类、聚类、关联规则挖掘、时间序列挖掘、回归等，其中分类、回归属于预测性任务，聚类、关联规则挖掘、时间序列分析等则都是解释性任务。



> 笔者初学



## DataScience LifeCycle

> [DataScience-Life-Cycle](https://github.com/okulbilisim/awesome-datascience/blob/master/DataScience-Life-Cycle.md)

DataScience是从数据中获取价值的最重要的流行方法之一。

### Identify Problem(定位问题)

这个环节最重要的是需求和数据的匹配。首先需要明确需求，比如分类问题中需要判断是否有带标注的训练数据集，否则，无法按照有监督的分类算法来解决。此外，数据的规模、重要Feature的覆盖率等等，也都是要考虑的问题。



### Data PreProcess(数据预处理)

####  数据集成、数据冗余与数值冲突

数据挖掘中准备数据的时候，需要尽可能的将相关数据集成在一起。如果集成的数据中，有两列或者多列值一样，则不可避免地会产生数值冲突或者数据冗余，可能需要根据数据的质量来决定保留冲突中的哪一列。

#### 数据采样

一般来说，有效的采样方式如下：如果样本是有代表性的，则使用样本数据和使用整个数据集的效果几乎是一样的。抽样方法有很多，首先考虑的是有放回的采样，以及具体选择哪种采样方式。

#### 数据清洗、缺失值处理与噪声数据

现实世界中的数据，是真实的数据，不可避免地会存在各种各样异常的情况。比如某一列的值缺失或者某列的值是异常的。所以，我们需要在数据预处理阶段进行数据清洗，来减少噪音数据对于模型训练和预测结果的影响。

### Feature Engineering(特征工程)

数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限。

> feature engineering is another topic which doesn’t seem to merit any review papers or books, or even chapters in books, but it is absolutely vital to ML success. […] Much of the success of machine learning is actually success in engineering features that a learner can understand.

特征是对于所需解决问题有用的属性。比如在计算机视觉领域，图片作为研究对象，可能图片中的一个线条就是一个特征；在自然语言处理领域，研究对象是文档，文档中一个词语出现的次数就是一个特征；在语音识别领域，研究对象是一段话，那么一个音频位就是一个特征。

#### 特征的提取、选择和构造

既然特征是对我们所解决的问题最有用的属性，首先我们需要处理的是根据原始数据抽取出所需要的特征。不过需要注意的是并不是所有的特征对所解决的问题影响一样大，有些特征可能对问题产生特别大的影响，但是有些则影响甚微，和所解决的问题不相关的特征需要被剔除掉。因此，我们需要针对所解决的问题选择最有用的特征集合，一般可以通过相关系数等方式来计算特征的重要性。

当然，有些模型本身会输出feature重要性，譬如Random Forest算法，而对于图片、音频等原始数据形态特别大的对象，则可能需要采用像PCA这样的自动降维技术。另外，还可能需要本人对数据和所需解决的问题有深入的理解，能够通过特征组合等方式构造出新的特征，这也正是特征工程被称为一门艺术的原因。

### 算法和模型

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



## Reference

### Practices & Resources


- [DataScience/MachineLearning Toolkits Index:笔者总结的数据科学/机器学习中常用工具集合/示例索引]()

- [awesome-datascience](https://github.com/okulbilisim/awesome-datascience)

- [data-science-ipython-notebooks](https://github.com/donnemartin/data-science-ipython-notebooks)

### Blogs & News

- [Data science blogs](https://github.com/rushter/data-science-blogs)


### Books & Tools

- [部分免费的DataScience方面的书籍](http://www.learndatasci.com/free-books/)

- [面向程序员的数据挖掘指南](https://github.com/egrcc/guidetodatamining)

- [Introduction to Machine Learning with Python](https://github.com/amueller/introduction_to_ml_with_python)

- [A Programmer's Guide to Data Mining](http://guidetodatamining.com/):一本开源的介绍了数据挖掘方面的书，包括了推荐系统、协同过滤、分类、Native Bayes、基本的NLP以及聚类



# DataMining

## 方法论

### 机器学习：从样本中学习的智能程序

在90年代初，人们开始意识到一种可以更有效地构建模式识别算法的方法，那就是用数据（可以通过廉价劳动力采集获得）去替换专家（具有很多图像方面知识的人）。因此，我们搜集大量的人脸和非人脸图像，再选择一个算法，然后冲着咖啡、晒着太阳，等着计算机完成对这些图像的学习。这就是机器学习的思想。“机器学习”强调的是，在给计算机程序（或者机器）输入一些数据后，它必须做一些事情，那就是学习这些数据，而这个学习的步骤是明确的。相信我，就算计算机完成学习要耗上一天的时间，也会比你邀请你的研究伙伴来到你家然后专门手工得为这个任务设计一些分类规则要好。

![典型的机器学习流程][1]

### 深度学习：一统江湖的架构

深度学习强调的是你使用的模型（例如深度卷积多层神经网络），模型中的参数通过从数据中学习获得。然而，深度学习也带来了一些其他需要考虑的问题。因为你面对的是一个高维的模型（即庞大的网络），所以你需要大量的数据（大数据）和强大的运算能力（图形处理器，GPU）才能优化这个模型。卷积被广泛用于深度学习（尤其是计算机视觉应用中），而且它的架构往往都是非浅层的。

![ConvNet框架][2]

> [Hacker's guide to Neural Networks][3]

## 应用

### 模式识别：智能程序的诞生

模式识别是70年代和80年代非常流行的一个术语。它强调的是如何让一个计算机程序去做一些看起来很“智能”的事情，例如识别“3”这个数字。而且在融入了很多的智慧和直觉后，人们也的确构建了这样的一个程序。例如，区分“3”和“B”或者“3”和“8”。早在以前，大家也不会去关心你是怎么实现的，只要这个机器不是由人躲在盒子里面伪装的就好（图2）。不过，如果你的算法对图像应用了一些像滤波器、边缘检测和形态学处理等等高大上的技术后，模式识别社区肯定就会对它感兴趣。光学字符识别就是从这个社区诞生的。因此，把模式识别称为70年代，80年代和90年代初的“智能”信号处理是合适的。决策树、启发式和二次判别分析等全部诞生于这个时代。而且，在这个时代，模式识别也成为了计算机科学领域的小伙伴搞的东西，而不是电子工程。从这个时代诞生的模式识别领域最著名的书之一是由Duda & Hart执笔的“模式识别（Pattern Classification）”。对基础的研究者来说，仍然是一本不错的入门教材。不过对于里面的一些词汇就不要太纠结了，因为这本书已经有一定的年代了，词汇会有点过时。

![模式识别示范][4]
