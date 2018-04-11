### Python 深度学习框架回顾

### 摘要

本文翻译自 Madison May 发布的 [Python Deep Learning Frameworks Reviewed](https://indico.io/blog/python-deep-learning-frameworks-reviewed/)，经原作者授权由 InfoQ 中文站翻译并分享。本文对于常用的基于 Python 的深度学习框架 Theano、 Lasagne、 Blocks、 TensorFlow、Keras、MXNet、PyTorch 进行了介绍与优劣比较，有助于深度学习入门者对于这些框架形成初步的认识。

### 正文

最近我一直在思考某个 Data Science Stack Exchange 上存在很久的问题：[什么才是最好的基于 Python 的神经网络库？](http://datascience.stackexchange.com/a/695/684) 时光飞逝，在过去的两年半的时间里涌现出了很多基于 Python 的 深度学习框架，而我 2014 年七月份推荐的 pylearn2 这个框架早已物是人非，不再维护。可喜的是已经有不少优秀的深度学习框架填补了它的空缺，成为了 indico 日常产品开发中重要的组成部分。当然，尺有所短，寸有所长，每个框架都有其优势与不足；我也希望在本文中基于自己的工作经历对于 2017 年中的 Python 深度学习生态进行一个综合宏观的介绍，希望为初学者勾勒出一幅清晰的群雄逐鹿图。具体而言，本文会着眼于以下框架：

* Theano
* Lasagne
* Blocks
* TensorFlow
* Keras
* MXNet
* PyTorch

### [Theano](https://github.com/Theano/Theano)

#### **描述：**

Theano 是一个允许你定义、优化以及高效执行包含多维数组的数学表达式的 Python 库；它支持 GPU 操作，并且能够进行高速的符号微分运算。

#### **文档：**

[http://deeplearning.net/software/theano/](http://deeplearning.net/software/theano/)

#### **总结：**

Theano 不仅仅是一个可以独立使用的库，它还是我们下面介绍的很多框架的底层数值计算引擎；它来自蒙特利尔大学 MILA 实验室，由 Frédéric Bastien 最早创建。Theano 提供的 API 相对底层，因此如果你希望高效运行 Theano， 你必须对它的底层算法非常熟悉。如果你拥有丰富的机器学习理论知识与经验，并且你希望对于自己的模型有细粒度的控制或者自己动手创建新的模型，那么 Theano 是个不错的选择。总结而言，Theano 最大的优势就是其灵活性。

#### **优势：**

* 相对灵活
* 正确使用的话性能较好

#### **不足：**

* 陡峭的学习曲线
* 大量的底层 API
* 编译复杂符号图的时候可能会很慢

#### **资源：**

* [安装指南](http://deeplearning.net/software/theano/install.html)
* [官方教程](http://deeplearning.net/software/theano/tutorial/)
* [Theano 幻灯介绍与实践案例](https://github.com/goodfeli/theano_exercises)
* [基于 Theano 实现从线性拟合到 CNN](https://github.com/Newmu/Theano-Tutorials)
* [基于 Python & Theano 的深度学习介绍](https://indico.io/blog/introduction-to-deep-learning-with-python-and-theano-2/)

---

### [Lasagne](https://github.com/Lasagne/Lasagne)

#### **描述：**

基于 Theano 的轻量级神经网络构建与训练库。

#### **文档：**

[http://lasagne.readthedocs.org/](http://lasagne.readthedocs.org/)

#### **总结：**

鉴于 Theano 着重打造面向符号数学的工具库，Lasagne 提供了基于 Theano 的相对高层的抽象，使它对于偏向工程的深度学习开发者更为友好。它最早由 DeepMind 的研究学者 Sander Dieleman 开发与维护。不同于 Theano 中网络模型需要指定为符号变量的表达式，Lasagne 允许用户以层的概念来定义网络，并且引入了所谓的 “Conv2DLayer” 与 “DropoutLayer”。Lasagne 以牺牲部分灵活性为代价提供了常用的组件来进行层构建、初始化、模型正则化、模型监控与模型训练。

#### **优势：**

* 还是比较灵活的
* 比 Theano 提供了更高层的抽象接口
* 文档与代码更为条理清晰

#### **不足：**

* 社区并不是很活跃

#### **资源：**

* [官方 GitHub 主页](https://github.com/Lasagne/Lasagne)
* [官方 安装 指南](http://lasagne.readthedocs.io/en/latest/user/installation.html)
* [官方 Lasagne 教程](http://lasagne.readthedocs.io/en/latest/user/tutorial.html)
* [示范代码](https://github.com/Lasagne/Lasagne/tree/master/examples)

---

### [Blocks](https://github.com/mila-udem/blocks)

#### **描述：**

基于 Theano 的神经网络构建与训练框架。

#### **文档：**

[http://blocks.readthedocs.io/en/latest/](http://blocks.readthedocs.io/en/latest/)

#### **总结：**

类似于 Lasagne，Blocks 在 Theano 的基础上添加了对于层的抽象，从而方便定义更加清晰，简单且标准的深度学习模型。它同样来源于蒙特利尔大学的 MILA 实验室，同样也是 Theano 与 PyLearn2 的作者。Blocks 相对于 Lasagne 具有更复杂的学习曲线，不过它也提供了更加灵活的接口操作。除此之外，Blocks 对于循环神经网络架构有非常好的支持，因此我个人很推荐你尝试下它；Blocks 也是除了 TensorFlow 之外我们在 indico 工作中首先考虑的选择了。

#### **优势：**

* 还是比较灵活的
* 比 Theano 提供了更高层的抽象接口
* 提供了相对完备的测试

#### **不足：**

* 陡峭的学习曲线
* 社区不是很活跃

#### **资源：**

* [官方安装指南](http://blocks.readthedocs.io/en/latest/setup.html)
* [Blocks 库构建的 Arxiv 论文](https://arxiv.org/pdf/1506.00619.pdf)
* [Reddit 上关于 Blocks 与 Lasagne 区别的讨论](https://www.reddit.com/r/MachineLearning/comments/4kpztm/lasagne_vs_blocks_for_deep_learning/)
* [Block 的姐妹库 Fuel](https://github.com/mila-udem/fuel)

---

### [TensorFlow](https://github.com/tensorflow/tensorflow)

#### **描述：**

使用数据流图进行数值计算的开源软件库。

#### **文档：**

[https://www.tensorflow.org/api_docs/python/](https://www.tensorflow.org/api_docs/python/)

#### **总结：**

TensorFlow 集成了类似于 Theano 这样底层的符号计算功能，也包含了类似于 Blocks 或者 Lasagne 这样的高层 API。尽管 TensorFlow 登上 Python 深度学习库的时间尚短，但是它已经成为了最受瞩目、社区最为庞大的工具。TensorFlow 由 Google 大脑团队发布并且维护，它支持多 GPU 的机器学习模型，提供了高效的数据管道以及内建的用于审视、可视化以及序列化模型的功能。最近 TensorFlow 团队决定添加对于 Keras 的内建支持，使得 TensorFlow 具备更好的可用性。尽管社区都认同 [TensorFlow 是有缺陷的](https://indico.io/blog/the-good-bad-ugly-of-tensorflow/)，但是因为其社区的庞大与项目背后支持力量，学习 TensorFlow 会是个不错的选择；TensorFlow 也是现在 indico 的首选机器学习库。

#### **优势：**

* 由 Google 支持
* 社区很繁荣
* 同时提供了底层与高层的网络训练接口
* 比 Theano 能够更快地训练模型
* 清晰的多 GPU 支持

#### **不足：**

* 一开始的时候性能不是很好
* 对 RNN 的支持度仍然落后于 Theano

#### **资源：**

* [官方 TensorFlow 主页](https://www.tensorflow.org/)
* [下载与安装指南](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md)
* [indico 对于 TensorFlow 的评价](https://indico.io/blog/the-good-bad-ugly-of-tensorflow/)
* [一系列 TensorFlow 指南](https://github.com/nlintz/TensorFlow-Tutorials)
* [Udacity 上关于 TensorFlow 的教程](https://www.udacity.com/course/deep-learning--ud730)
* [TensorFlow MNIST 指南](https://www.tensorflow.org/tutorials/mnist/beginners/)
* [TensorFlow 数据输入](https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/)

---

### [Keras](https://github.com/fchollet/keras)

#### **描述：**

支持卷积神经网络，循环神经网络的 Python 深度学习库，能够运行在 Theano 或者 TensorFlow 之上。

#### **文档：**

[https://keras.io/](https://keras.io/)

#### **总结：**

Keras 算是这个列表中提供了最高层接口、用户使用最友好的深度学习库了。它由 Google 大脑团队的 Francis Chollet 创建与维护；它允许用户自由选择底层模型构建框架，可以是 Theano 或者 TensorFlow。Keras 的用户交互借鉴了 Torch，如果你有基于 Lua 进行机器学习的经验，Keras 会是很值得一试的工具。因为 Keras 完善的文档与简单易用的接口，Keras 的社区非常繁荣与活跃。最近，TensorFlow 团队宣布计划将内建支持 Keras，因此不久的将来 Keras 会是 TensorFlow 项目的子集了吧。

#### **优势：**

* 你可以自由选择使用 Theano 或者 TensorFlow
* 直观，高级的接口
* 相对简单的学习曲线

#### **不足：**

* 与其他相比灵活性略差

#### **资源：**

* [官方安装指南](https://keras.io/#installation)
* [Keras 用户的 Google 社群](https://groups.google.com/forum/#!forum/keras-users)
* [Keras 使用示例](https://github.com/fchollet/keras/tree/master/examples)
* [基于 Docker 的 Keras 环境搭建](https://github.com/fchollet/keras/tree/master/docker)
* [Keras 教程](https://github.com/fchollet/keras-resources)

---

### [MXNet](https://github.com/dmlc/mxnet)

#### **描述：**

MXNet 致力于提供兼顾性能与灵活性的深度学习框架。

#### **文档：**

[http://mxnet.io/api/python/index.html#python-api-reference](http://mxnet.io/api/python/index.html#python-api-reference)

#### **总结：**

作为 Amazon 的钦定深度学习框架，MXNet 也算是性能最好的深度学习框架之一了。它提供了类似于 Theano 与 TensorFlow 的数据流图，并且支持多 GPU 配置 ，提供了类似于 Lasagne 与 Blocks 的相对高阶的模型构建块，还能运行在多种硬件设备上（包括移动设备）。Python 只是 MXNet 支持的多种语言之一，它还提供了基于 R, Julia, C++, Scala, Matlab 以及 JavaScript 的多种接口。如果你专注于效率，那么 MXNet 是个不二选择，不过你可能会要让自己习惯 MXNet 中很多的奇怪设定。

#### **优势：**

* 相当快的评测结果
* 彻底的灵活性

#### **不足：**

* 社区最小
* 比 Theano 更陡峭的学习曲线

#### **资源：**

* [官方初始化指南](http://mxnet.io/get_started/)
* [indico 对于 MXNet 的介绍](https://indico.io/blog/getting-started-with-mxnet/)
* [MXNet 示例仓库](https://github.com/dmlc/mxnet/tree/master/example)
* [Amazon 的 CTO 对于 MXNet 的评价](http://www.allthingsdistributed.com/2016/11/mxnet-default-framework-deep-learning-aws.html)
* [MXNet Arxiv 论文](https://arxiv.org/abs/1512.01274)

---

### [PyTorch](https://github.com/pytorch/pytorch)

#### **描述：**

支持强力 GPU 加速的 Python Tensor 与 动态神经网络库。

#### **文档：**

[http://pytorch.org/docs/](http://pytorch.org/docs/)

#### **总结：**

PyTorch 问世不过数周，在我们的深度学习框架列表中尚属新生儿。虽然 PyTorch 主要基于 Lua Torch， 但是它是由 Facebook 人工智能研究团队（FAIR）支持的，并且它设计初衷就是用来处理动态计算图问题，这个特性也是其他的 Theano，TensorFlow，以及其他扩展框架所没有的。虽然 PyTorch 尚未成熟，但是因为它这一特性我们相信它会在未来的 Python 深度学习生态系统中占据一席之地，并且我们认为它是个非常不错的选项。

#### **优势：**

* 由 Facebook 支持与维护
* 支持动态图计算
* 同时提供了高层接口与底层接口

#### **不足：**

* 与竞争者相比还不成熟
* 除了官方文档之外的相关资料尚处于短缺

#### **资源：**

* [官方主页](http://pytorch.org/)
* [PyTorch twitter](https://twitter.com/PyTorch)
* [PyTorch 使用示例仓库](https://github.com/pytorch/examples)
* [PyTorch 教程仓库](https://github.com/pytorch/tutorials)

> 参考英文原文：[Python Deep Learning Frameworks Reviewed](https://indico.io/blog/python-deep-learning-frameworks-reviewed/)
