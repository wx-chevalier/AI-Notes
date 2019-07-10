# 激活函数

## 激活函数

构建神经网络中非常重要的一个环节就是选择合适的激活函数\(Activation Function\)，激活函数是为了增加神经网络模型的非线性，也可以看做从数据空间到最终表达空间的一种映射。全连接层只是对数据做仿射变换（affine transformation），而多个仿射变换的叠加仍然是一个仿射变换。解决问题的一个方法是引入非线性变换，例如对隐藏变量使用按元素运算的非线性函数进行变换，然后再作为下一个全连接层的输入。

仅就 sigmod 与 tahn 相比时，在大部分情况下我们应该优先使用 tahn 函数；除了在最终的输出层，因为输出层我们需要得到的是 0~1 范围内的概率表示。譬如在上面介绍的浅层神经网络中，我们就可以使用 sigmod 作为隐层的激活函数，而使用 tahn 作为输出层的激活函数。

不过 sigmod 与 tahn 同样都存在在极大值或者极小值处梯度较小、收敛缓慢的问题。并且采用 sigmoid 等函数，算激活函数时\(指数运算\)，计算量大，反向传播求误差梯度时，求导涉及除法，计算量相对大；而采用 ReLU\(rectified linear unit\) 激活函数，整个过程的计算量节省很多。此外，ReLU 会使一部分神经元的输出为 0，这样就造成了网络的稀疏性，并且减少了参数的相互依存关系，缓解了过拟合问题的发生。

## ReLU 函数

ReLU（rectified linear unit）函数提供了一个很简单的非线性变换。给定元素 $x$ ，该函数定义为：

$$
\operatorname{ReLU}(x)=\max (x, 0)
$$

![](https://i.postimg.cc/Pfzk1wK6/image.png)

显然，当输入为负数时，ReLU 函数的导数为 0；当输入为正数时，ReLU 函数的导数为 1。尽管输入为 0 时 ReLU 函数不可导，但是我们可以取此处的导数为 0。

## sigmoid 函数

sigmoid 函数可以将元素的值变换到 0 和 1 之间：

$$
\operatorname{sigmoid}(x)=\frac{1}{1+\exp (-x)}
$$

sigmoid 函数在早期的神经网络中较为普遍，但它目前逐渐被更简单的 ReLU 函数取代。

![](https://i.postimg.cc/k4nYPfRz/image.png)

依据链式法则，sigmoid 函数的导数为：

$$
\text { sigmoid }^{\prime}(x)=\operatorname{sigmoid}(x)(1-\operatorname{sigmoid}(x))
$$

## tanh 函数

tanh（双曲正切）函数可以将元素的值变换到-1 和 1 之间：

$$
\tanh (x)=\frac{1-\exp (-2 x)}{1+\exp (-2 x)}
$$

我们接着绘制 tanh 函数。当输入接近 0 时，tanh 函数接近线性变换。虽然该函数的形状和 sigmoid 函数的形状很像，但 tanh 函数在坐标系的原点上对称。

![](https://i.postimg.cc/15LW39Np/image.png)

依据链式法则，tanh 函数的导数：

$$
\tanh ^{\prime}(x)=1-\tanh ^{2}(x)
$$

