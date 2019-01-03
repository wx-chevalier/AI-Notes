# 凸集基本概念

![](http://7xkt0f.com1.z0.glb.clouddn.com/D3D1E50E-F7FB-456A-89CB-18D9D0BABA15.png)

如上图所示，$y=x^2$是一个凸函数，函数图像上位于$y=x^2$上方的区域构成凸集。凸函数图像的上方区域，一定是凸集；一个函数图像的上方区域为凸集，则该函数是凸函数。

### 直线的向量表达

已知二维平面上两个定点$A(4,1)$、$B(1,3)$，那么经过$A、B$两点的直线方程为：

$$
\left\{
\begin{aligned}
x_1 = \theta *4 + (1-\theta)*1 \\
x_2 = \theta _ 1 + (1-\theta) _ 3\\
\end{aligned}
\theta \in R  
\right.
$$

那么对应的向量形式就是：$\vec x = \theta*\vec a + (1-\theta)*\vec b$

那么推广来说，几何体的向量表达如下所示：

![](http://7xiegq.com1.z0.glb.clouddn.com/7C219296-C270-48F2-B8E0-88C1E64BE63E.png)

### 仿射集(Affine set)

通过集合$C$中任意两个不同点的直线仍然在集合$C$内，则称集合$C$为仿射集。

$$

\forall \theta \in R, \forall x_1,x_2 \in C , 则 x=\theta x_1 + (1 - \theta)x_2 \in C

$$

### 凸集

集合$C$内任意两点间的线段均在集合$C$内，则称集合$C$为凸集。

$$

\forall \theta \in [0,1], \forall x_1,x_2 \in C , 则 x=\theta x_1 + (1 - \theta)x_2 \in C

$$

如果转化为$k$个点的版本，即

$$
\forall x*1, \dots, x_k \in C, \theta_i \in [0,1]，且 \sum*{i=1}^k \theta*i = 1，则\sum*{i=1}^k \theta_ix_i = C
$$

因为仿射集的条件比凸集条件强，所以，仿射集必然是凸集。

![](http://7xlgth.com1.z0.glb.clouddn.com/9C4568AA-80A7-46AD-8D95-E00A10CD22D4.png)

### 凸包

集合$C$的所有点的凸组合形成的集合，叫做集合$C$的凸包。

$$
conv C = \{\sum*{i=1}^k \theta_ix_i|x_i \in C, \theta_i \ge 0, \sum*{i=1}^k\theta_i = 1 \}
$$

集合$C$的凸包是能够包含$C$的最小的凸集。

## 超平面和半空间

超平面 hyperplane 的定义如下：

$\{x|a^Tx=b\}$

半空间 half space 定义如下：

$\{x|a^Tx \le b\} \{x|a^Tx \ge b\}$

- 欧式球

$$
B(x_c,r)=\{ x| \Arrowvert x-x_c\Arrowvert_2 \le r \} \\
= \{ x| (x-x_c)^T(x-x_c) \le r^2 \}
$$

- 椭球

$E={x|(x-x_c)^TP(x-x_c) \le r^2}$

- 范数球

$B(x_c,r)=\{x| \Arrowvert x-x_c\Arrowvert \le r\}$

- 范数锥

$\{(x,t)|\Arrowvert x \Arrowvert \le t\}$

### 多面体

多面体即为有限个半空间和超平面的交集

$P={x|a^T_jx \le b_j,c_i^Tx=d_i}$

仿射集(如超平面、直线)、射线、线段、半空间都是多面体，多面体肯定是一个凸集，此外，有界的多面体有时称作多胞形(polytope)。

![](http://7xkt0f.com1.z0.glb.clouddn.com/71096C98-8525-4CBA-92F3-CF07D8581871.png)

### 分割超平面

假设$C$和$D$为两不相交的凸集，则存在超平面$P$，$P$可以将$C$和$D$分离：

$\forall x \in C, a^Tx \le b 且 \forall x \in D, a^Tx \ge b$

![](http://7xlgth.com1.z0.glb.clouddn.com/6F7D0227-1B4F-4C9B-9D6A-7D2F064ACB06.png)

两个集合的距离，定义为两个集合间元素的最短距离，然后做集合 C 和集合 D 最短线段的垂直平分线，即得到了最优的超平面：

![](http://7xlgth.com1.z0.glb.clouddn.com/DC5C76BB-7194-40EE-AF2E-EB9F010CF460.png)

### 支撑超平面

假设集合$C$，$x0$为$C$边界上的点。如果存在$a \ne 0$，满足对$\forall x \in C $，都有$a^Tx \le a^Tx_0$成立，则称超平面${x|a^Tx = a^Tx_0}$为集合$C$在点$x0$处的支撑超平面。

![](http://7xlgth.com1.z0.glb.clouddn.com/07D4E1B4-89D9-4084-BE34-27C8114A040D.png)

## 运算

### 保持凸性的运算

- 集合交运算：两个凸集的集合也是凸集

* 仿射交换

​ 函数$f=Ax+b$的形式，称函数是仿射的：即线性函数加常数的形式

- 透视变换

* 投射变换(线性分式变换)

#### 集合交运算：半空间的交

#### 仿射变换

$f(x)=Ax+b,A \in R^{m\*n},b \in R^m$

伸缩、平移、投影

若$f$是仿射变换，$f:R^n \to R^m f(S)=\{f(x)|x \in S\}$

如果$S$为凸集，则$f(S)$为凸集。

两个凸集的和为凸集、两个凸集的笛卡尔积为凸集。

#### 透视变换

透视函数对向量进行伸缩(规范化)，使得最后一维的分量为 1 并舍弃。

$P:R^{n+1} \to R^n, P(z,t) = z/t$

透视的直观意义就是小孔成像。

凸集的透视变换的结果还是一个凸集。

#### 投射函数(线性分式函数)

投射函数是透视函数和仿射函数的复合。$g$为仿射函数：

$$
g:R^n \to R^{n+1} \\
g(x) =
\begin{bmatrix} A \\ c^T \end{bmatrix}x + \begin{bmatrix} b \\ d \end{bmatrix}
A \in R^{m\*n}, b \in R^{m}, c \in R^n, d \in R
$$

定义$f$为线性分式函数

$f(x)=(Ax+b)/(c^Tx + d)，dome = \{ x | c^Tx +d > 0 \}$

其中，如果$c=0,d>0$，则$f$即为普通的仿射函数。

# 凸函数基本概念

## 凸函数

若函数$f$的定义域$domf$为凸集，且满足$\forall x,y \in dom f, 0 \le \theta \le 1$，有

$f(\theta x + (1- \theta)y) \le \theta f(x) + (1-\theta)f(y)$

![](http://7xlgth.com1.z0.glb.clouddn.com/B771D505-1979-4D5B-827C-FC56BF0F9CB2.png)

若$f$一阶可微，则函数$f$为凸函数的条件是当且仅当$f$的定义域$domf$为凸集，且

$\forall x,y \in dom f, f(y) \ge f(x) + \bigtriangledown f(x)^T(y-x)$

![](http://7xlgth.com1.z0.glb.clouddn.com/1A0FDF4A-9573-4B41-A921-0A12A4D3A536.png)

若函数$f$二阶可微，则函数$f$为凸函数当且仅当$domf$为凸集，且

$\bigtriangledown^2f(x) \ge 0$

- 若$f$是一元函数，上式表示二阶导大于等于 0
- 若$f$是多元函数，上式表示二阶导 Hessian 矩阵半正定

注意，根据定义可知，直线也是凸函数。

#### 上镜图

函数$$f$$的图像定义为$$\{(x,f(x))|x \in dom f \}$$，函数$$f$$的上镜图(epigraph)定义为：

$$epi f = \{(x,t) | x \in dom f, f(x) \le t\}$$

![](http://7xkt0f.com1.z0.glb.clouddn.com/C79C0496-E217-4C2A-B71A-F2DD81FD7B48.png)

一个函数是凸函数，当且仅当其上镜图是凸集。反之，如果一个函数是凹函数，当且仅当其亚图是凸集。

$$hypo f = \{(x,t)|t \le f(x)\}$$

### Jensen 不等式

在$$f$$是凸函数的情况下，基本的 Jensen 不等式为:$$f(\theta x + (1-\theta)y) \le \theta f(x) + (1 - \theta)f(y)$$

如果是多维不等式，即如果$$\theta_1, \dots, \theta_k \ge 0, \theta_1 + \dots + \theta_k = 1$$，则

$$f(\theta_1x_1 + \dots + \theta_kx_k) \le \theta_1 f(x_1) + \dots + \theta_k f(x_k) $$

若$p(x) \ge 0 \quad on \quad S \subseteq dom f , \int_Sp(x)dx = 1$

则 $$f(\int_Sp(x)xdx) \le \int_Sf(x)p(x)dx$$，也就是$$f(Ex) \le Ef(x)$$

Jensen 不等式是几乎所有不等式的基础。

#### 保持函数凸性的算子

- 凸函数的非负加权和

$$f(x) = \omega_1f_1(x)+ \dots + \omega_nf_n(x)$$

- 凸函数与仿射函数的复合

$$g(x)=f(Ax+b)$$

- 凸函数的逐点最大值、逐点上确界

$$f(x)=max(f*1(x),\dots,f_n(x)) \\ f(x)=sup*{y \in A}g(x,y)$$

### 共轭函数

原函数$$f:R^n \to R$$共轭函数定义：

$$f^\*(y) = sup\_{x \in dom f}(y^Tx-f(x))$$

显然，定义式的右端是关于$$y$$的仿射函数，它们逐点求上确界，得到的函数$$f^\*(y)$$一定是凸函数。凸函数的共轭函数的共轭函数就是其本身。

#### Fenchel 不等式

根据共轭函数的定义，立刻可以得到：

$$f(x) + f^\*(y) \ge x^Ty$$

# 凸优化

优化问题的基本形式：

$$
minimize \quad f_0(x), x \in R^n \\
subject \quad to \quad \\
f_i(x) \le 0, i = 1,\dots,m \\
h_j(x) = 0, j = 1,\dots,p \\
优化变量 x \in R^n \\
不等式约束 f_i(x) \le 0
等式约束 h_j(x) = 0
无约束优化 m=p=0
$$

而凸优化中，即限制条件为$$f_i(x)$$为凸函数，$$h_j(x)$$为仿射函数。凸优化问题的重要性质即凸优化问题的可行域为凸集，凸优化问题的局部最优解即为全局最优解。

## 对偶问题

对于上面所说的凸优化的基本问题，可以带入 Lagrange 函数：

$$L(x,\lambda,v) = f*0(x) + \sum^m*{i=1}\lambda*if_i(x) + \sum*{j=1}^pv_jh_j(x)$$

对于固定的$$x$$，Lagrange 函数即为关于$$\lambda$$和$$v$$的仿射函数。

然后对于该函数求下界：

$$g(\lambda,v)=inf*{x \in D}L(x,\lambda,v)=inf*{x \in D}( f*0(x) + \sum^m*{i=1}\lambda*if_i(x) + \sum*{j=1}^pv_jh_j(x))$$

如果没有下确界，定义：$$g(\lambda,v) = - \infty$$。根据定义，显然有对于$$\forall \lambda > 0, \forall v$$，若原优化问题有最优值$$p^_$$，则有$$g(\lambda,v) \le p^_$$，进一步可以得到 Lagrange 对偶函数为凸函数。

![](http://7xkt0f.com1.z0.glb.clouddn.com/CD0DB9A2-80D4-4B33-9454-948123AE8EE1.png)

上图左侧为原函数，右侧为对偶函数。
