# TensorFire：浏览器端的 TensorFlow

编译：王下邀月熊

------

## 摘要：

TensorFire 是基于 WebGL 的运行在浏览器内的高性能神经网络框架，其执行速度甚至可以快于原生的 TensorFlow。

## 正文：

深度学习与人工智能技术正在逐步地改变人们的生活，以 TensoFlow 为代表的一系列深度学习与神经网络框架也是如日中天，迅猛发展。TensorFire 则是基于 WebGL 的，运行在浏览器中的神经网络框架；使用 TensorFire 编写的应用能够在实现前沿深度学习算法的同时，不需要任何的安装或者配置就直接运行在现代浏览器中。与之前某些浏览器内的神经网络框架相比，TensorFire 有着近百倍的速度提升，甚至于能够与那些运行在本地 CPU 上的代码性能相媲美。现代的 PC、笔记本电脑与移动终端往往都被包含能够进行高性能并发计算的 GPU，通过将神经网络中的权重转化为 WebGL 中的纹理，TensorFire 将神经网络中的层转化为了片段着色器（Fragment Shaders），从而利用原本设计来加速执行 3D 游戏的引擎来执行神经网络。另一方面，不同于其他的 WebGL 计算框架，TensorFire 支持 Low-precision Quantized Tensors，从而保证了模型的适用性。

TensorFire 主要由两部分组成：底层基于 GLSL 的能够高效编写操作四维张量的并行 WebGLS 着色器的编程语言，以及上层的用于导入 Keras 与 TensorFlow 训练好的模型的接口。TensorFire 能够运行在任何的，无论是否支持 CUDA 的 GPU 上；这就意味着，譬如最新的 2016 Retina MacBook Pro 这样的使用 AMD 显卡的机器，也能顺畅地运行 TensorFire。TensorFire 能够帮助开发者构建不需要用户本地安装的智能应用，并且不同于传统的收集用户数据以统一训练的模式，直接将模型下发到用户端能够保障用户隐私权。TensorFire 官方正在着手提供多个范例，譬如复杂的 ResNet-152 网络、著名的基于 RNN 的文本生产与图片着色、基于 SqueeseNet 的物体识别与分类等等。开发者也可以使用 TensorFire 提供的底层接口来进行其他的高性能计算，譬如 PageRank、元胞自动机仿真、图片转化与过滤等等。

TensorFire 项目由多位 MIT 的毕业生协作而成。其中 Kevin Kwok 与 Guillermo Webster 曾编写过[ Project Naptha ](https://projectnaptha.com/)这样的将 JavaScript 与计算机视觉相结合的从图片中提取文字的 OCR 项目。Anish Athalye 与 Logan Engstrom 则编写过首个[ Gatys' Neural Artistic Style ](https://github.com/anishathalye/neural-style) 以及[ Johnson's Fast Style Transfer ](https://github.com/lengstrom/fast-style-transfer)算法的 TensorFlow 模型。

该项目 Style Transfer Neural Network Demo 链接：[https://tenso.rs/demos/fast-neural-style/](https://tenso.rs/demos/fast-neural-style/)

查看英文原文： [TensorFire](https://tenso.rs/#wat)