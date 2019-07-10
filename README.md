# Introduction

![default](https://user-images.githubusercontent.com/5803001/44629093-c753d900-a97c-11e8-8c16-9d0e96b149aa.png)

## 人工智能与深度学习实战

在本系列中，你可能会接触到数据挖掘、机器学习、深度学习、自然语言处理、人工智能等很多的概念。最后，如果你想了解数据结构与传统算法，那么可以参考 [数据结构与算法 https://url.wx-coder.cn/S84SI](https://url.wx-coder.cn/S84SI)。

值得说明的是，本系列文章的配套代码归纳于 [AIDL-Workbench](https://github.com/wx-chevalier/AIDL-Workbench)，特别是工具实践篇中的大部分内容是以 Juypter Notebook 的形式放在该仓库中，强烈建议按照目录层级对照浏览相关内容。

You may find the structure of this book loose, deliberately. Because the definition of Data Science is vague.

## Navigation \| 导航

您可以通过以下任一方式阅读笔者的系列文章，涵盖了技术资料归纳、编程语言与理论、Web 与大前端、服务端开发与基础架构、云计算与大数据、数据科学与人工智能、产品设计等多个领域：

* 在 Gitbook 中在线浏览，每个系列对应各自的 Gitbook 仓库。

| [Awesome Lists](https://ngte-al.gitbook.io/i/) | [Awesome CheatSheets](https://ngte-ac.gitbook.io/i/) | [Awesome Interviews](https://github.com/wx-chevalier/Developer-Zero-To-Mastery/tree/master/Interview) | [Awesome RoadMaps](https://github.com/wx-chevalier/Developer-Zero-To-Mastery/tree/master/RoadMap) | [Awesome-CS-Books-Warehouse](https://github.com/wx-chevalier/Awesome-CS-Books-Warehouse) |
| :--- | :--- | :--- | :--- | :--- |


| [编程语言理论与实践](https://ngte-pl.gitbook.io/i/) | [软件工程、数据结构与算法、设计模式、软件架构](https://ngte-se.gitbook.io/i/) | [现代 Web 开发基础与工程实践](https://ngte-web.gitbook.io/i/) | [大前端混合开发与数据可视化](https://ngte-fe.gitbook.io/i/) | [服务端开发实践与工程架构](https://ngte-be.gitbook.io/i/) | [分布式基础架构](https://ngte-infras.gitbook.io/i/) | [数据科学，人工智能与深度学习](https://ngte-aidl.gitbook.io/i/) | [产品设计与用户体验](https://ngte-pd.gitbook.io/i/) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |


* 前往 [xCompass https://wx-chevalier.github.io](https://wx-chevalier.github.io/home/#/search) 交互式地检索、查找需要的文章/链接/书籍/课程，或者关注微信公众号：某熊的技术之路。

![](https://i.postimg.cc/3RVYtbsv/image.png)

* 在下文的 [MATRIX 文章与代码矩阵 https://github.com/wx-chevalier/Developer-Zero-To-Mastery](https://github.com/wx-chevalier/Developer-Zero-To-Mastery) 中查看文章与项目的源代码。

| [数理统计](https://github.com/wx-chevalier/AIDL-Series/tree/263727d4092380f0c07c3a640796d17aed503601/数理统计/README.md) | [数据分析](https://github.com/wx-chevalier/AIDL-Series/tree/263727d4092380f0c07c3a640796d17aed503601/数据分析/README.md) | [机器学习](https://github.com/wx-chevalier/AIDL-Series/tree/263727d4092380f0c07c3a640796d17aed503601/机器学习/README.md) | [深度学习](https://github.com/wx-chevalier/AIDL-Series/tree/263727d4092380f0c07c3a640796d17aed503601/深度学习/README.md) | [自然语言处理](https://github.com/wx-chevalier/AIDL-Series/tree/263727d4092380f0c07c3a640796d17aed503601/自然语言处理/README.md) | [推荐系统等行业应用](https://github.com/wx-chevalier/AIDL-Series/tree/263727d4092380f0c07c3a640796d17aed503601/行业应用/README.md) | [课程笔记](https://github.com/wx-chevalier/AIDL-Series/tree/263727d4092380f0c07c3a640796d17aed503601/课程笔记/README.md) | [TensorFlow & PyTorch 等工具实践](https://github.com/wx-chevalier/AIDL-Series/tree/263727d4092380f0c07c3a640796d17aed503601/工具实践/README.md) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |


## Preface \| 前言

1956 年，几个计算机科学家相聚在达特茅斯会议，提出了“人工智能”的概念，梦想着用当时刚刚出现的计算机来构造复杂的、拥有与人类智慧同样本质特性的机器。2012 年以后，得益于数据量的上涨、运算力的提升和机器学习新算法（深度学习）的出现，人工智能开始大爆发。

![](https://i.postimg.cc/26CpMVQK/image.png)

机器学习是一种实现人工智能的方法，机器学习最基本的做法，是使用算法来解析数据、从中学习，然后对真实世界中的事件做出决策和预测。与传统的为解决特定任务、硬编码的软件程序不同，机器学习是用大量的数据来“训练”，通过各种算法从数据中学习如何完成任务。机器学习直接来源于早期的人工智能领域，传统的模型算法包括决策树、聚类、贝叶斯分类、支持向量机、EM、Adaboost 等等。从任务类型上来分，机器学习算法可以分为监督学习（如分类问题）、无监督学习（如聚类问题）、半监督学习、集成学习和强化学习的等。传统的机器学习算法在指纹识别、基于 Haar 的人脸检测、基于 HoG 特征的物体检测等领域的应用基本达到了商业化的要求或者特定场景的商业化水平，但每前进一步都异常艰难，直到深度学习算法的出现。

深度学习是一种实现机器学习的技术，深度学习本来并不是一种独立的学习方法，其本身也会用到有监督和无监督的学习方法来训练深度神经网络。最初的深度学习是利用深度神经网络来解决特征表达的一种学习过程。深度神经网络本身并不是一个全新的概念，可大致理解为包含多个隐含层的神经网络结构。为了提高深层神经网络的训练效果，人们对神经元的连接方法和激活函数等方面做出相应的调整。其实有不少想法早年间也曾有过，但由于当时训练数据量不足、计算能力落后，因此最终的效果不尽如人意。深度学习摧枯拉朽般地实现了各种任务，使得似乎所有的机器辅助功能都变为可能。无人驾驶汽车，预防性医疗保健，甚至是更好的电影推荐，都近在眼前，或者即将实现。

不同的模型、策略、算法的搭配，不断地推动着人工智能的发展，其又可以被分为三个阶段：计算智能、感知智能和认知智能。

* 第一阶段的计算智能即快速计算和记忆存储，像机器人战胜围棋大师，靠的就是超强的记忆能力和运算速度。人脑的逻辑能力再强大，也敌不过人工智能每天和自己下几百盘棋，通过强大的计算能力对十几步后的结果做出预测，从这一角度来说，人工智能多次战败世界级围棋选手，足以证明这一领域发展之成熟。
* 第二阶段的感知智能，即让机器拥有视觉、听觉、触觉等感知能力。自动驾驶汽车做的就是这一方面的研究，使机器通过传感器对周围的环境进行感知和处理，从而实现自动驾驶。感知智能方面的技术目前发展比较成熟的领域有语音识别和图像识别，比如做安全领域人脸识别技术的 Face++，以成熟的计算机视觉技术深耕电商、短视频等领域的 Yi+，能够对多种语言进行准确识别翻译的科大讯飞等。
* 第三阶段的认知智能就是让机器拥有自己的认知，能理解会思考。认知智能是目前机器和人差距最大的领域，因为这不仅涉及逻辑和技术，还涉及心理学、哲学和语言学等学科。

最后，我们通过几张全景图来了解我们在本系列中会学习哪些知识：

![](https://i.postimg.cc/L8w4YDPd/image.png)

![](https://i.postimg.cc/GhMmrdhm/image.png)

![](https://i.postimg.cc/pLpXL4pY/image.png)

## 版权

![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg) ![](https://parg.co/bDm)

笔者所有文章遵循[知识共享 署名-非商业性使用-禁止演绎 4.0 国际许可协议](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.zh)，欢迎转载，尊重版权。

![default](https://i.postimg.cc/y1QXgJ6f/image.png)

