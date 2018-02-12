# PCA
PCA（Principal Component Analysis）不仅仅是对高维数据进行降维，更重要的是经过降维去除了噪声，发现了数据中的模式。PCA把原先的n个特征用数目更少的m个特征取代，新特征是旧特征的线性组合，这些线性组合最大化样本方差，尽量使新的m个特征互不相关。从旧特征到新特征的映射捕获数据中的固有变异性。
根据上面对PCA的数学原理的解释，我们可以了解到一些PCA的能力和限制。PCA本质上是将方差最大的方向作为主要特征，并且在各个正交方向上将数据“离相关”，也就是让它们在不同正交方向上没有相关性。
因此，PCA也存在一些限制，例如它可以很好的解除线性相关，但是对于高阶相关性就没有办法了，对于存在高阶相关性的数据，可以 考虑Kernel PCA，通过Kernel函数将非线性相关转为线性相关，关于这点就不展开讨论了。另外，PCA假设数据各主特征是分布在正交方向上，如果在非正交方向上 存在几个方差较大的方向，PCA的效果就大打折扣了。
最后需要说明的是，PCA是一种无参数技术，也就是说面对同样的数据，如果不考虑清洗，谁来做结果都一样，没有主观参数的介入，所以PCA便于通用实现，但是本身无法个性化的优化。
希望这篇文章能帮助朋友们了解PCA的数学理论基础和实现原理，借此了解PCA的适用场景和限制，从而更好的使用这个算法。


## Reference
- [主成分分析（PCA）原理详解](http://blog.jobbole.com/109015/)


PCA算法
总结一下PCA的算法步骤：
设有m条n维数据。
1）将原始数据按列组成n行m列矩阵X
2）将X的每一行（代表一个属性字段）进行零均值化，即减去这一行的均值
3）求出协方差矩阵$C=\frac{1}{m}XX^{T}$
4）求出协方差矩阵的特征值及对应的特征向量
5）将特征向量按对应特征值大小从上到下按行排列成矩阵，取前k行组成矩阵P
6）$Y=PX$即为降维到k维后的数据

这里以上文提到的
\begin{pmatrix} -1 & -1 & 0 & 2 & 0 \\ -2 & 0 & 0 & 1 & 1 \end{pmatrix}
为例，我们用PCA方法将这组二维数据其降到一维。
因为这个矩阵的每行已经是零均值，这里我们直接求协方差矩阵：
C=\frac{1}{5}\begin{pmatrix} -1 & -1 & 0 & 2 & 0 \\ -2 & 0 & 0 & 1 & 1 \end{pmatrix}\begin{pmatrix} -1 & -2 \\ -1 & 0 \\ 0 & 0 \\ 2 & 1 \\ 0 & 1 \end{pmatrix}=\begin{pmatrix} \frac{6}{5} & \frac{4}{5} \\ \frac{4}{5} & \frac{6}{5} \end{pmatrix}
然后求其特征值和特征向量，具体求解方法不再详述，可以参考相关资料。求解后特征值为：
\lambda_1=2,\lambda_2=2/5
其对应的特征向量分别是：
c_1\begin{pmatrix} 1 \\ 1 \end{pmatrix},c_2\begin{pmatrix} -1 \\ 1 \end{pmatrix}
其中对应的特征向量分别是一个通解，c_1和c_2可取任意实数。那么标准化后的特征向量为：
\begin{pmatrix} 1/\sqrt{2} \\ 1/\sqrt{2} \end{pmatrix},\begin{pmatrix} -1/\sqrt{2} \\ 1/\sqrt{2} \end{pmatrix}
因此我们的矩阵P是：
P=\begin{pmatrix} 1/\sqrt{2} & 1/\sqrt{2} \\ -1/\sqrt{2} & 1/\sqrt{2} \end{pmatrix}
可以验证协方差矩阵C的对角化：
PCP^\mathsf{T}=\begin{pmatrix} 1/\sqrt{2} & 1/\sqrt{2} \\ -1/\sqrt{2} & 1/\sqrt{2} \end{pmatrix}\begin{pmatrix} 6/5 & 4/5 \\ 4/5 & 6/5 \end{pmatrix}\begin{pmatrix} 1/\sqrt{2} & -1/\sqrt{2} \\ 1/\sqrt{2} & 1/\sqrt{2} \end{pmatrix}=\begin{pmatrix} 2 & 0 \\ 0 & 2/5 \end{pmatrix}
最后我们用P的第一行乘以数据矩阵，就得到了降维后的表示：
Y=\begin{pmatrix} 1/\sqrt{2} & 1/\sqrt{2} \end{pmatrix}\begin{pmatrix} -1 & -1 & 0 & 2 & 0 \\ -2 & 0 & 0 & 1 & 1 \end{pmatrix}=\begin{pmatrix} -3/\sqrt{2} & -1/\sqrt{2} & 0 & 3/\sqrt{2} & -1/\sqrt{2} \end{pmatrix}
降维投影结果如下图：
![](http://blog.codinglabs.org/uploads/pictures/pca-tutorial/07.png)
