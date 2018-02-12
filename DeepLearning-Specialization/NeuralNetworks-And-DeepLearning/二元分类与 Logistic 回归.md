
> [Andrew NG 深度学习课程笔记：二元分类与 Logistic 回归](https://zhuanlan.zhihu.com/p/28530027)从属于笔者的[Deep Learning Specialization 课程笔记](https://parg.co/bjz)系列文章，本文主要记述了笔者学习 Andrew NG [Deep Learning Specialization](https://www.coursera.org/learn/neural-networks-deep-learning/) 系列课程的笔记与代码实现。注意，本篇有大量的数学符号与表达式，部分网页并不支持；可以前往[源文件](https://parg.co/b25)查看较好的排版或者在自己的编辑器中打开。


# 二元分类与 Logistic 回归


本部分将会介绍神经网格构建与训练的基础知识；一般来说，网络的计算过程由正向传播（Forward Propagation）与反向传播（Back Propagation）两部分组成。这里我们将会以简单的 Logistic 回归为例，讲解如何解决常见的二元分类（Binary Classification）问题。这里我们将会尝试训练出简单的神经网络以自动识别某个图片是否为猫，为猫则输出 1，否则输出 0。计算机中的图片往往表示为红、绿、蓝三个通道的像素值；如果我们的图像是 64 * 64 像素值大小，我们的单张图片的特征维度即为 64 * 64 * 3 = 12288，即可以使用 $n_x = 12288$ 来表示特征向量的维度。


![](https://coding.net/u/hoteam/p/Cache/git/raw/master/2017/8/2/WX20170814-203645.png)


# 深度学习的标准术语约定


## 神经网络的符号


上标 $^{(i)}$ 表示第 $i$ 个训练用例，而上标 $^{[l]}$ 则表示第 $l$ 层。


### 尺寸
- $m$：数据集中用例的数目。
- $n_x$：输入的单个用例的特征向量维度。
- $n_y$：输出的维度（待分类的数目）。
- $n_h^{[l]}$：第 $l$ 个隐层的单元数目，在循环中，我们可能定义 $n_x = n_h^{[0]}$ 以及 $n_y = n_h^{number \, of \, layers + 1}$。
- $L$：神经网络的层数。


### 对象
- $X \in R^{n_x \times m}$：输入的矩阵，即包含了 $m$ 个用例，每个用例的特征向量维度为 $n_x$。
- $x^{(i)} \in R^{n_x}$：第 $i$ 个用例的特征向量，表示为列向量。
- $Y \in R^{n_y \times m}$：标签矩阵。
- $y^{(i)} \in R^{n_y}$：第 $i$ 个用例的输出标签。
- $W^{[l]} \in R^{number \, of \, units \, in \, next \, layer \times number \, of \, unites \, in \, the \, previous \, layer}$：第 $l$ 层与第 $l+1$ 层之间的权重矩阵，在简单的二元分类且仅有输入层与输出层的情况下，其维度就是 $ 1 \times n_x$。
- $b^{[l]} \in R^{number \, of \, units \, in \, next \, layer}$：第 $l$ 层的偏差矩阵。
- $\hat{y} \in R^{n_y}$：输出的预测向量，也可以表示为 $a^{[L]}$，其中 $L$ 表示网络中的总层数。


### 通用前向传播等式


- $ a = g^{[l]}(W_xx^{(i)} + b_1) = g^{[l]}(z_1) $，其中 $g^{[l]}$ 表示第 $l$ 层的激活函数。
- $\hat{y}^{(i)} = softmax(W_hh + b_2)$。
- 通用激活公式：$a_j^{[l]} = g^{[l]}(\sum_kw_{jk}^{[l]}a_k^{[l-1]} + b_j^{[l]}) = g^{[l]}(z_j^{[l]})$。
- $J(x, W, b, y)$ 或者 $J(\hat{y}, y)$ 表示损失函数。


### 损失函数


- $J_{CE(\hat{y},y)} = - \sum_{i=0}^m y^{(i)}log \hat{y}^{(i)}$
- $J_{1(\hat{y},y)} = \sum_{i=0}^m | y^{(i)} - \hat{y}^{(i)} |$


## 深度学习的表示
在深度学习中，使用结点代表输入、激活函数或者数据，边代表权重或者偏差，下图即是两个典型的神经网络：
![](https://coding.net/u/hoteam/p/Cache/git/raw/master/2017/8/2/WX20170814-211522.png) 
![](https://coding.net/u/hoteam/p/Cache/git/raw/master/2017/8/2/WX20170814-211546.png) 


# Logistic 回归


## 基础模型


在猫咪识别问题中，我们给定了未知的图片，可以将其表示为 $X \in R^{n_x}$ 的特征向量；我们的任务就是寻找合适的算法，来根据特征向量推导出该图片是猫咪的概率。在上面的介绍中我们假设了 Logistic 函数的参数为 $w \in R^{n_x} $ 以及 $b \in R$，则输出的计算公式可以表示为：
$$
\hat{y} = \sigma(w^Tx + b)
$$
这里的 $\sigma$ 表示 Sigmoid 函数，该函数的表达式与线型如下：


![](https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Sigmoid-function-2.svg/2000px-Sigmoid-function-2.svg.png)


上图中可以发现，当 $t$ 非常大时，$e^{-t}$ 趋近于 0，整体的函数值趋近于 1；反之，如果 $t$ 非常小的时候，整体的函数值趋近于 0。


## 损失函数与代价函数


我们的训练目标是在给定训练数据 $\{(x^{(1)}, y^{(1)}),...,(x^{(m)},y^{(m)})\}$ 的情况下使得 $\hat{y}^{(i)}$ 尽可能接近 $y^{(i)}$，而所谓的损失函数即是用于衡量预测结果与真实值之间的误差。最简单的损失函数定义方式为平方差损失：
$$
L(\hat{y},y) = \frac{1}{2} (\hat{y} - y)^2
$$
不过 Logistic 回归中我们并不倾向于使用这样的损失函数，因为其对于梯度下降并不友好，很多情况下会陷入非凸状态而只能得到局部最优解。这里我们将会使用如下的损失函数：
$$

L(\hat{y},y) = -(ylog\hat{y} + (1-y)log(1-\hat{y}))
$$
我们的优化目标是希望损失函数值越小越好，这里我们考虑两个极端情况，当 $y = 1$ 时，损失函数值为 $-log\hat{y}$；此时如果 $\hat{y} = 1$，则损失函数为 0。反之如果 $\hat{y} = 0$，则损失函数值趋近于无穷大。当 $y = 0$ 时，损失函数值为 $-log(1-\hat{y})$；如果 $\hat{y} = 1$，则损失函数值也变得无穷大。这样我们可以将 Logistic 回归中总的代价函数定义为：
$$
J(w,b) = 
\frac{1}{m}\sum_{i=1}^mL(\hat{y}^{(i)} - y^{(i)}) =
-\frac{1}{m} \sum_{i=1}^m [y^{(i)}log\hat{y}^{(i)} + (1-y^{(i)})log(1-\hat{y}^{(i)})]
$$
在深度学习的模型训练中我们常常会接触到损失函数（Loss Function）与代价函数（Cost Function）的概念，其中损失函数代指单个训练用例的错误程度，而代价函数往往是整个训练集中所有训练用例的损失函数值的平均。





