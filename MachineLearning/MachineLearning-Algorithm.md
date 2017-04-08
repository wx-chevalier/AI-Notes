
# Introduction

## Definition(机器学习定义)

对于某给定的任务T，在合理的性能度量方案P的前提下，某计算机程序可以自主学习任务T的经验E；随着提供合适、优质、大量的经验E，该程序对于任务T的性能逐步提高。即随着任务的不断执行，经验的累积会带来计算机性能的提升。

## Algorithms

> - [a-tour-of-machine-learning-algorithms](http://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/)

算法基本上从功能或者形式上来分类。比如，基于树的算法，神经网络算法。这是一个很有用的分类方式，但并不完美。因为有许多算法可以轻易地被分到两类中去，比如说Learning Vector Quantization就同时是神经网络类的算法和基于实例的方法。正如机器学习算法本身没有完美的模型一样，算法的分类方法也没有完美的。

### Grouped By Learning Style(根据学习风格分类)

#### Supervised Learning

输入数据被称为训练数据，并且有已知的结果或被标记。比如说一封邮件是否是垃圾邮件，或者说一段时间内的股价。模型做出预测，如果错了就会被修正，这个过程一直持续到对于训练数据它能够达到一定的正确标准。问题例子包括分类和回归问题，算法例子包括逻辑回归和反向神经网络。

![](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Supervised-Learning-Algorithms.png)

有监督学习的通俗例子可以以人类认知月亮的过程为例，小时候会认为黑夜里天空中明亮的圆盘状物体为月亮，后来随着时间的推移，发现圆盘状这个特点并非一个决定性因素。

#### [Unsupervised Learning](https://github.com/okulbilisim/awesome-datascience/blob/master/Algorithms.md#unsupervised-learning)

输入数据没有被标记，也没有确定的结果。模型对数据的结构和数值进行归纳。问题例子包括Association rule learning和聚类问题，算法例子包括 Apriori 算法和K-均值算法。

![](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Unsupervised-Learning-Algorithms.png)

#### Semi-Supervised Learning

![](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Semi-supervised-Learning-Algorithms.png)

输入数据是被标记的和不被标记的数据的混合，有一些预测问题但是模型也必须学习数据的结构和组成。问题例子包括分类和回归问题，算法例子基本上是无监督学习算法的延伸。


### Grouped By Model Construction(根据模型构建分类)

> [discriminative-modeling-vs-generative-modeling](http://freemind.pluskid.org/machine-learning/discriminative-modeling-vs-generative-modeling/)



机器学习的算法或者模型的分类有很多种，其中有一种分法把模型分为 Discriminative Modeling （判别模型）和 Generative Modeling （生成模型）两种。为了写起来方便我们以下简称 DM 和 GM 。在一个基本的机器学习问题中，我们通常有输入 $x \in X$和输出 $y \in Y$两个部分，简单地来说，DM 关注于 $x$和 $y$的关系，或者说在给定某个 $x$的情况下所对应的 $y$应该满足的规律或分布，即评估对象是最大化条件概率$p(y|x)$并直接对其建模；而 GM 则试图描述 $x$和 $y$的联合分布，即模型评估对象是最大化联合概率$p(x,y)$并对其建模。


   其实两者的评估目标都是要得到最终的类别标签$Y$， 而$Y=argmax p(y|x)$，不同的是判别式模型直接通过解在满足训练样本分布下的最优化问题得到模型参数，主要用到拉格朗日乘算法、梯度下降法，常见的判别式模型如最大熵模型、CRF、LR、SVM等；

   而生成式模型先经过贝叶斯转换成$Y = argmax p(y|x) = argmax p(x|y)*p(y)$，然后分别学习$p(y)$和$p(x|y)$的概率分布，主要通过极大似然估计的方法学习参数，如NGram、HMM、Naive Bayes。

    比较而言，判别式模型会更加灵活，而生成式模型一般需要将特征加入马尔科夫链。但是判别式模型需要有指导训练，而生成式模型可以无指导训练。







### Grouped By Similarity(根据算法相似度分类)

#### Regression Algorithms

![Regression Algorithms](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Regression-Algorithms.png)

Regression is concerned with modelling the relationship between variables that is iteratively refined using a measure of error in the predictions made by the model.

Regression methods are a workhorse of statistics and have been cooped into statistical machine learning. This may be confusing because we can use regression to refer to the class of problem and the class of algorithm. Really, regression is a process.

The most popular regression algorithms are:

- Ordinary Least Squares Regression (OLSR)
- Linear Regression
- Logistic Regression
- Stepwise Regression
- Multivariate Adaptive Regression Splines (MARS)
- Locally Estimated Scatterplot Smoothing (LOESS)

#### Instance-based Algorithms

![Instance-based Algorithms](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Instance-based-Algorithms.png)Instance based learning model a decision problem with instances or examples of training data that are deemed important or required to the model.

Such methods typically build up a database of example data and compare new data to the database using a similarity measure in order to find the best match and make a prediction. For this reason, instance-based methods are also called winner-take-all methods and memory-based learning. Focus is put on representation of the stored instances and similarity measures used between instances.

The most popular instance-based algorithms are:

- k-Nearest Neighbour (kNN)
- Learning Vector Quantization (LVQ)
- Self-Organizing Map (SOM)
- Locally Weighted Learning (LWL)

#### Regularization Algorithms

![Regularization Algorithms](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Regularization-Algorithms.png)

An extension made to another method (typically regression methods) that penalizes models based on their complexity, favoring simpler models that are also better at generalizing.

I have listed regularization algorithms separately here because they are popular, powerful and generally simple modifications made to other methods.

The most popular regularization algorithms are:

- Ridge Regression
- Least Absolute Shrinkage and Selection Operator (LASSO)
- Elastic Net
- Least-Angle Regression (LARS)

#### Decision Tree Algorithms

![Decision Tree Algorithms](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Decision-Tree-Algorithms.png)Decision tree methods construct a model of decisions made based on actual values of attributes in the data.

Decisions fork in tree structures until a prediction decision is made for a given record. Decision trees are trained on data for classification and regression problems. Decision trees are often fast and accurate and a big favorite in machine learning.

The most popular decision tree algorithms are:

- Classification and Regression Tree (CART)
- Iterative Dichotomiser 3 (ID3)
- C4.5 and C5.0 (different versions of a powerful approach)
- Chi-squared Automatic Interaction Detection (CHAID)
- Decision Stump
- M5
- Conditional Decision Trees

#### Bayesian Algorithms

![Bayesian Algorithms](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Bayesian-Algorithms.png)Bayesian methods are those that are explicitly apply Bayes’ Theorem for problems such as classification and regression.

The most popular Bayesian algorithms are:

- Naive Bayes
- Gaussian Naive Bayes
- Multinomial Naive Bayes
- Averaged One-Dependence Estimators (AODE)
- Bayesian Belief Network (BBN)
- Bayesian Network (BN)

#### Clustering Algorithms

![Clustering Algorithms](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Clustering-Algorithms.png)Clustering, like regression describes the class of problem and the class of methods.

Clustering methods are typically organized by the modelling approaches such as centroid-based and hierarchal. All methods are concerned with using the inherent structures in the data to best organize the data into groups of maximum commonality.

The most popular clustering algorithms are:

- k-Means
- k-Medians
- Expectation Maximisation (EM)
- Hierarchical Clustering

#### Artificial Neural Network Algorithms

![Artificial Neural Network Algorithms](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Artificial-Neural-Network-Algorithms.png)

Artificial Neural Networks are models that are inspired by the structure and/or function of biological neural networks.

They are a class of pattern matching that are commonly used for regression and classification problems but are really an enormous subfield comprised of hundreds of algorithms and variations for all manner of problem types.

Note that I have separated out Deep Learning from neural networks because of the massive growth and popularity in the field. Here we are concerned with the more classical methods.

The most popular artificial neural network algorithms are:

- Perceptron
- Back-Propagation
- Hopfield Network
- Radial Basis Function Network (RBFN)

#### Deep Learning Algorithms

![Deep Learning Algorithms](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Deep-Learning-Algorithms.png)Deep Learning methods are a modern update to Artificial Neural Networks that exploit abundant cheap computation.

They are concerned with building much larger and more complex neural networks, and as commented above, many methods are concerned with semi-supervised learning problems where large datasets contain very little labelled data.

The most popular deep learning algorithms are:

- Deep Boltzmann Machine (DBM)
- Deep Belief Networks (DBN)
- Convolutional Neural Network (CNN)
- Stacked Auto-Encoders

#### Support Vector Machines

#### Ensemble Algorithms

![Ensemble Algorithms](http://3qeqpr26caki16dnhd19sv6by6v.wpengine.netdna-cdn.com/wp-content/uploads/2013/11/Ensemble-Algorithms.png)

Ensemble methods are models composed of multiple weaker models that are independently trained and whose predictions are combined in some way to make the overall prediction.

Much effort is put into what types of weak learners to combine and the ways in which to combine them. This is a very powerful class of techniques and as such is very popular.

- Boosting
- Bootstrapped Aggregation (Bagging)
- AdaBoost
- Stacked Generalization (blending)
- Gradient Boosting Machines (GBM)
- Gradient Boosted Regression Trees (GBRT)
- Random Forest



组合方式主要分为Bagging和Boosting。Bagging是Bootstrapped Aggregation的缩写，其基本原理是让学习算法训练多轮，每轮的训练集由从初始的训练集中随机取出的N个训练样本组成（有放回的随机抽样），训练之后可得到一个预测函数集合，通过投票的方式决定预测结果。而Boosting中主要的是AdaBoost(Adaptive Boosting)。基本原理是初始化时对每一个训练样本赋相等的权重$1/n$，然后用学习算法训练多轮，每轮结束后，对训练失败的训练样本赋以较大的权重，也就是让学习算法在后续的学习中集中对比较难的训练样本进行学习，从而得到一个预测函数的集合。每个预测函数都有一定的权重，预测效果好的函数权重较大，反之较小，最终通过有权重的投票方式来决定预测结果。而Bagging与Boosting的主要区别在于：

- 取样方式不同。Bagging采用均匀取样，而Boosting根据错误率来取样，因此理论上来说Boosting的分类精度要优于Bagging。

- 训练集的选择方式不同。Bagging的训练集的选择是随机的，各轮训练集之间相互独立，而Boosting的各轮训练集的选择与前面的学习结果有关。

- 预测函数不同。Bagging的各个预测函数没有权重，而Boosting是有权重的。Bagging的各个预测函数可以并行生成，而Boosting的各个预测函数只能顺序生成。



总结而言，对于像神级网络这样及其耗时的学习方法，Bagging可以通过并行训练节省大量时间的开销。Bagging和Boosting都可以有效地提高分类的准确性。在大多数数据集合中，Boosting的准确性比Bagging要高。



## 模型求解

### 参数估计

给定某系统的若干样本，求取该系统的参数，主要分为频率学派与贝叶斯学派。

#### 频率学派

假定参数是某个或者某些未知的定值，求这些参数如何取值，能够使得某目标函数取得极大或者极小值，典型的代表有矩估计、MLE、MaxEnt以及EM。

#### 贝叶斯学派

基于贝叶斯模型，假设参数本身是变化的，服从某个分布。求在这个分布约束下使得某目标函数极大/极小。

大数据下是频率学派对于贝叶斯学派的一次有效逆袭。


## Reference

### Tutorials & Docs

- [机器学习温和指南](http://www.csdn.net/article/2015-09-08/2825647#rd?sukey=b0cb5c5b9e50130317c130ba2243b8e573404f264fe1467cde82d02d2529978108ee91bb9a2bbc0d81118e2db77390ca)
- [机器学习与数据挖掘的学习路线图](http://mp.weixin.qq.com/s?__biz=MzA3MDg0MjgxNQ==&mid=2652389718&idx=1&sn=9f0c3a6f525f4c28a504e7475efc8f02&scene=0#wechat_redirect&utm_source=tuicool&utm_medium=referral)
- [dive-into-machine-learning](https://github.com/hangtwenty/dive-into-machine-learning)
- [Stanford-UFLDL教程](http://ufldl.stanford.edu/wiki/index.php/UFLDL%E6%95%99%E7%A8%8B)
- [机器学习视频教程——小象训练营，这里面提供了一系列的课程的说明](http://www.chinahadoop.cn/course/423)



### Books & Tools

- [Standford-The Elements of Statistical Learning Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/local.ftp/Springer/OLD/ESLII_print4.pdf)

### Resources & Practices

- [a-tour-of-machine-learning-algorithms](http://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/)

- [史上最全的机器学习资料](https://yq.aliyun.com/articles/43089?utm_source=tuicool&utm_medium=referral)

### Blogs & News

- [zouxy09的专栏](http://blog.csdn.net/zouxy09)
- [一个机器学习算法的博客](http://www.algorithmdog.com/)


# 范数与规则化

> [机器学习中的范数规则化](http://blog.csdn.net/zouxy09/article/details/24971995)

机器学习中最常见的问题就是过拟合与规则化的问题，本节即是对于这两个问题进行些许的阐述。

































