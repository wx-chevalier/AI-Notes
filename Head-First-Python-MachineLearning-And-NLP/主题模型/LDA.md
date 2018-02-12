# Latent Dirichlet 
# Dirichlet Distribution & Dirichlet Process:狄利克雷分布于狄利克雷过程

作者：肉很多链接：https://www.zhihu.com/question/26751755/answer/80931791来源：知乎著作权归作者所有，转载请联系作者获得授权。要想易懂地理解dirichlet distribution，首先先得知道它的特殊版本beta distribution干了什么。而要理解beta distribution有什么用，还得了解Bernoulli process。

首先先看**Bernoulli process**。要理解什么是Bernoulli process，首先先看什么Bernoulli trial。Bernoulli trial简单地说就是一个只有两个结果的简单trial，比如**\*抛硬币***。
那我们就用**抛一个(不均匀）硬币**来说好了，X = 1就是头，X = 0就是字，我们设定q是抛出字的概率。
那什么是bernoulli process？就是从Bernoulli population里随机抽样，或者说就是重复的独立Bernoulli trials，再或者说就是狂抛这枚硬币n次记结果吧（汗=_=）。好吧，我们就一直抛吧，我们记下X=0的次数k.

现在问题来了。
Q：**我们如何知道这枚硬币抛出字的概率？**我们知道，如果可以一直抛下去，最后k/n一定会趋近于q；可是现实中有很多场合不允许我们总抛硬币，比如**我只允许你抛4次**。你该怎么回答这个问题？显然你在只抛4次的情况下，k/n基本不靠谱；那你只能"**猜一下q大致分布在[0,1]中间的哪些值里会比较合理**",但绝不可能得到一个准确的结果比如q就是等于k/n。

举个例子，比如：4次抛掷出现“头头字字”，你肯定觉得q在0.5附近比较合理，q在0.2和0.8附近的硬币抛出这个结果应该有点不太可能，q = 0.05和0.95那是有点扯淡了。
你如果把这些值画出来，你会发现q在[0,1]区间内呈现的就是一个中间最高，两边低的情况。从感性上说，这样应当是比较符合常理的。

那我们如果有个什么工具能描述一下这个q可能的分布就好了，比如用一个概率密度函数来描述一下? 这当然可以，可是我们还需要注意另一个问题，那就是随着n增长观测变多，**你每次的概率密度函数该怎么计算**？该怎么利用以前的结果更新（这个在形式上和计算上都很重要）？

到这里，其实很自然地会想到把bayes theorem引进来，因为Bayes能随着不断的观测而更新概率；而且每次只需要前一次的prior等等…在这先不多说bayes有什么好，接下来用更形式化语言来讲其实说得更清楚。

**我们现在用更正规的语言重新整理一下思路。**现在有个硬币得到random sample X  = (x1,x2,...xn)，我们需要基于这n次观察的结果来估算一下**q在[0,1]中取哪个值比较靠谱**，由于我们不能再用单一一个确定的值描述q，所以我们用一个分布函数来描述：有关q的概率密度函数（说得再简单点，即是q在[0,1]“分布律”）。当然，这应当写成一个条件密度：f(q|X)，因为我们总是观测到X的情况下，来猜的q。

现在我们来看看Bayes theorem，看看它能带来什么不同：
![P(q|x) P(x) = P(X=x|q)P(q)](//zhihu.com/equation?tex=P%28q%7Cx%29+P%28x%29+%3D+P%28X%3Dx%7Cq%29P%28q%29)

在这里P(q)就是关于q的先验概率（所谓先验，就是在得到观察X之前，我们设定的关于q的概率密度函数）。P(q|x)是观测到x之后得到的关于q的后验概率。注意，到这里公式里出现的都是"概率"，并没有在[0,1]上的概率密度函数出现。为了让贝叶斯定理和密度函数结合到一块。我们可以从方程两边由P(q)得到f(q)，而由P(q|x)得到f(q|x)。
又注意到P(x)可以认定为是个常量（Q：why？），可以在分析这类问题时不用管。**那么，这里就有个简单的结论——****关于q的后验概率密度f(q|x)就和“关于q的****先验概率密度乘以一个条件概率"成比例，即：**
![f(q|x)\sim P(X=x|q)f(q)](//zhihu.com/equation?tex=f%28q%7Cx%29%5Csim+P%28X%3Dx%7Cq%29f%28q%29)

带着以上这个结论，我们再来看这个抛硬币问题：
连续抛n次，即为一个bernoulli process，则在q确定时，n次抛掷结果确定时，又观察得到k次字的概率可以描述为：![P(X=x|p) = q^{k}(1-q)^{n-k} ](//zhihu.com/equation?tex=P%28X%3Dx%7Cp%29+%3D+q%5E%7Bk%7D%281-q%29%5E%7Bn-k%7D+)
那么f(q|x)就和先验概率密度乘以以上的条件概率是成比例的：
![f(q|x) \sim q^{k}(1-q)^{n-k}f(q) ](//zhihu.com/equation?tex=f%28q%7Cx%29+%5Csim+q%5E%7Bk%7D%281-q%29%5E%7Bn-k%7Df%28q%29+)
虽然我们不知道，也求不出那个P(x)，但我们知道它是固定的，我们这时其实已经得到了一个求f(q|x)的公式（只要在n次观测下确定了，f(q)确定了，那么f(q|x)也确定了)。

现在在来看f(q)。显然，在我们对硬币一无所知的时候，我们应当认为硬币抛出字的概率q有可能在[0,1]上任意处取值。f(q)在这里取个均匀分布的密度函数是比较合适的，即f(q) = 1 (for q in [0,1]) 。
有些同学可能发现了，这里面![f(q|x) \sim q^{k}(1-q)^{n-k}](//zhihu.com/equation?tex=f%28q%7Cx%29+%5Csim+q%5E%7Bk%7D%281-q%29%5E%7Bn-k%7D)，**那个![q^{k}(1-q)^{n-k}](//zhihu.com/equation?tex=q%5E%7Bk%7D%281-q%29%5E%7Bn-k%7D)乘上[0,1]的均匀分布不就是一个Beta distribution么**？
对，它就是一个Beta distribution。Beta distribution由两个参数alpha、beta确定；在这里对应的alpha等于k+1，beta等于n+1-k。而**均匀分布的先验密度函数，就是那个f(q)也可以被beta distribution描述**，这时alpha等于1，beta也等于1。

更有意思的是，当我们每多抛一次硬币，出现字时，我们只需要alpha = alpha + 1；出现头只需要beta = beta + 1。这样就能得到需要估计的概率密度f(q|x)…

其实之所以计算会变得这么简单，是因为被beta distribution描述的prior经过bayes formula前后还是一个beta distribution；这种不改变函数本身所属family的特性，叫**共轭(conjugate)**。

ok。讲到这你应该明白，对于有两个结果的重复Bernoulli trial，我们用beta prior/distribution就能解决。那么加入我们有n个结果呢？比如抛的是骰子？
这时候上面的Bernoulli trial就要变成有一次trial有k个可能的结果； Bernoulli distribution就变成multinomial distribution。而beta distribution所表述的先验分布，也要改写成一个多结果版本的先验分布。那就是dirichlet distribution。
均匀的先验分布Beta(1,1)也要变成k个结果的Dir(alpha/K)。dirichlet prior也有共轭的性质，所以也是非常好计算的。
简而言之，就是由2种外推到k种，而看待它们的视角并没有什么不同。
他们有着非常非常非常相似的形式。

**结论1：dirichlet distribution就是由2种结果bernoulli trial导出的beta distribution外推到k种的generalization**


```py
from scipy.stats import dirichlet, poisson
from numpy.random import choice
from collections import defaultdict


num_documents = 5
num_topics = 2
topic_dirichlet_parameter = 1 # beta
term_dirichlet_parameter = 1 # alpha
vocabulary = ["see", "spot", "run"]
num_terms = len(vocabulary) 
length_param = 10 # xi

term_distribution_by_topic = {} # Phi
topic_distribution_by_document = {} # Theta
document_length = {}
topic_index = defaultdict(list)
word_index = defaultdict(list)

term_distribution = dirichlet(num_terms * [term_dirichlet_parameter])
topic_distribution = dirichlet(num_topics * [topic_dirichlet_parameter])

# 遍历每个主题
for topic in range(num_topics):
    # 采样得出每个主题对应的词分布
    term_distribution_by_topic[topic] = term_distribution.rvs()[0]

# 遍历所有的文档
for document in range(num_documents):
    # 采样出该文档对应的主题分布
    topic_distribution_by_document[document] = topic_distribution.rvs()[0]
    topic_distribution_param = topic_distribution_by_document[document]
    # 从泊松分布中采样出文档长度
    document_length[document] = poisson(length_param).rvs()
    
    # 遍历整个文档中的所有词
    for word in range(document_length[document]):
        topics = range(num_topics)
        # 采样出某个生成主题
        topic = choice(topics, p=topic_distribution_param)
        topic_index[document].append(topic)
        # 采样出某个生成词
        term_distribution_param = term_distribution_by_topic[topic]
        word_index[document].append(choice(vocabulary, p=term_distribution_param))
```


如果还有困惑的同学可以参考如下 Python 代码：
```py
def perplexity(self, docs=None):
    if docs == None: docs = self.docs
    # 单词在主题上的分布矩阵
    phi = self.worddist()
    log_per = 0
    N = 0
    Kalpha = self.K * self.alpha
    //遍历语料集中的所有文档
    for m, doc in enumerate(docs):
        // n_m_z 为每个文档中每个主题的单词数，theta 即是每个单词出现的频次占比
        theta = self.n_m_z[m] / (len(self.docs[m]) + Kalpha)
        for w in doc:
            // numpy.inner(phi[:,w], theta) 即是某个出现的概率统计值
            log_per -= numpy.log(numpy.inner(phi[:,w], theta))
        N += len(doc)
    return numpy.exp(log_per / N)
```



# Introduction
> LDA has been widely used in textual analysis,


LDA是标准的词袋模型。
> [通俗理解LDA主题模型](http://blog.csdn.net/v_july_v/article/details/41209515)

LDA主要涉及的问题包括共轭先验分布、Dirichlet分布以及Gibbs采样算法学习参数。LDA的输入为文档数目$M$，词数目$V$(非重复的term)，主题数目$K$。
![](http://7xlgth.com1.z0.glb.clouddn.com/5C724613-24AC-4782-B1DB-E890B87885FF.png)

![](https://coding.net/u/hoteam/p/Cache/git/raw/master/2016/8/1/6D650E67-A400-416C-A1AB-8E864513ED05.png)

# Mathematics
## Beta分布:Dirichlet分布的基础

Beta分布的概率密度为：
$$
f(x) = 
\left \{ 
\begin{aligned} 
\frac{1}{B(\alpha,\beta)}x^{\alpha-1}(1-x)^{\beta-1}, x \in (0,1) \\
0,其他
\end{aligned} 
\right. 
$$
其中$$B(\alpha,\beta) = \int_0^1 x^{\alpha - 1}(1-x)^{\beta-1}dx=\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}$$
其中Gamma函数可以看做阶乘的实数域的推广：
$$
\Gamma(x) = \int_0^{\infty}t^{x-1}e^{-t}dt \Rightarrow \Gamma(n) = (n-1)! \Rightarrow B(\alpha,\beta)=\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}
$$
Beta分布的期望为：
$$E(X) = \frac{\alpha + \beta}{\alpha}$$
## Dirichlet分布:多项分布的共轭分布
Dirichlet分布实际上就是把:
$$
\alpha = \alpha_1 ,
\beta = \alpha_2 ,
x = x_1 ,
x - 1 = x_2
$$

$$
f(\vec{p} | \vec{\alpha}) = \left \{
\begin{aligned}
\frac{1}{\Delta(\vec{\alpha})} \prod_{k=1}^{K} p_k^{\alpha_k - 1} ,p_k \in (0,1) \\
0,其他
\end{aligned}
\right.
$$
可以简记为：
$$
Dir(\vec{p} | \vec{\alpha}) = \frac{1}{\Delta(\vec{\alpha})} \prod_{k=1}^{K} p_k^{\alpha_k - 1}
$$
其中
$$
\Delta(\vec{\alpha}) = \frac{ \prod_{k=1}^K \Gamma(\alpha_k)}{ \Gamma(\sum_{k=1}^{K}\alpha_k)}
$$
该部分在给定的$\vec{\alpha}$情况下是可以计算出来值的。假设给定的一篇文档有50个主题，那么$\vec{\alpha}$就是维度为50的向量。在没有任何先验知识的情况下，最方便的也是最稳妥的初始化就是将这个50个值设置为同一个值。

### Symmetric Dirichlet Distribution(对称Dirichlet分布)

一旦采取了对称的Dirichlet分布，因为参数向量中的所有值都一样，公式可以改变为：
$$
Dir(\vec{p} | \alpha,K) = \frac{1}{\Delta_K(\alpha)} \prod_{k=1}^{K} p_k^{\alpha - 1} \\
\Delta_K(\vec{\alpha}) =  \Gamma^K(\alpha){ \Gamma(K * \alpha)}
$$
而不同的$\alpha$取值，当$\alpha=1$时候，退化为均匀分布。当$\alpha>1$时候，$p_1 = p_2 = \dots = p_k$的概率增大。当$\alpha<1$时候，$p_1 = 1 , p_{非i} = 0$的概率增大。映射到具体的文档分类中，$\alpha$取值越小，说明各个主题之间的离差越大。而$\alpha$值越大，说明该文档中各个主题出现的概率约接近。

![](http://7xkt0f.com1.z0.glb.clouddn.com/5C6D7D8E-3B58-41BF-8E6F-DBD1F1AA2B7A.png)

在实际的应用中，一般会选用$1/K$作为$\alpha$的初始值。

# 模型解释

![](http://7xlgth.com1.z0.glb.clouddn.com/D73D69FE-BA28-4E66-871F-B594B4BEFC29.png)
上图的箭头指向即是条件依赖。
## Terminology
- 字典中共有$V$个不可重复的term，如果这些term出现在了具体的文章中，就是word。在具体的某文章中的word当然是可能重复的。
- 语料库(Corpus)中共有$m$篇文档，分别是$d_1,d_2,\dots,d_m$，每篇文章长度为$N_m$，即由$N_i$个word组成。每篇文章都有各自的主题分布，主题分布服从多项式分布，该多项式分布的参数服从Dirichlet分布，该Dirichlet分布的参数为$\vec{ \alpha }$。注意，多项分布的共轭先验分布为Dirichlet分布。
> 怎么来看待所谓的文章主题服从多项分布呢。你每一个文章等于多一次实验，$m$篇文档就等于做了$m$次实验。而每次实验中有$K$个结果，每个结果以一定概率出现。

- 一共涉及到$K$(值给定)个主题，$T_1,T_2,\dots,T_k$。每个主题都有各自的词分布，词分布为多项式分布，该多项式分布的参数服从Dirichlet分布，该Diriclet分布的参数为$\vec{\beta}$。注意，一个词可能从属于多个主题。

## 模型过程
$\vec{\alpha}$与$\vec{\beta}$为先验分布的参数，一般会实现给定。如取0.1的对称Dirichlet分布，表示在参数学习结束后，期望每个文档的主题不会十分集中。

（1）选定文档主题

（2）根据主题选定词

## 参数学习
给定一个文档集合，$w_{mn}$是可以观察到的已知变量，$\vec{\alpha}$与$\vec{\beta}$是根据经验给定的先验参数，其他的变量$z_{mn}$、$\vec{\theta}$、$\vec{\varphi}$都是未知的隐含变量，需要根据观察到的变量来学习估计的。根据上图，可以写出所有变量的联合分布：


### 似然概率
一个词$w_{mn}$(即word，可重复的词)初始化为一个词$t$(term/token，不重复的词汇)的概率是：
$$
p(w_{m,n}=t | \vec{\theta_m},\Phi) = \sum_{k=1}^K p(w_{m,n}=t | \vec{\phi_k})p(z_{m,n}=k|\vec{\theta}_m)
$$
上式即给定某个主题的情况下能够看到某个词的概率的总和。每个文档中出现主题$k$的概率乘以主题$k$下出现词$t$的概率，然后枚举所有主题求和得到。整个文档集合的似然函数为：
$$
p(W | \Theta,\Phi) = \prod_{m=1}^{M}p(\vec{w_m} | \vec{\theta_m},\Phi) = \prod_{m=1}^M \prod_{n=1}^{N_m}p(w_{m,n}|\vec{\theta_m},\Phi)
$$
# Gibbs Sampling
> 首先通俗理解一下，在某篇文档中存在着$N_m$个词，依次根据其他的词推算某个词来自于某个主题的概率，从而达到收敛。最开始的时候，某个词属于某个主题是随机分配的。Gibbs Sampling的核心在于找出某个词到底属于哪个主题。

Gibbs Sampling算法的运行方式是每次选取概率向量的一个维度，给定其他维度的变量值采样当前度的值，不断迭代直到收敛输出待估计的参数。初始时随机给文本中的每个词分配主题$z^{(0)}$，然后统计每个主题$z$下出现词$t$的数量以及每个文档$m$下出现主题$z$的数量，每一轮计算$p(z_i|z_{\neq i},d,w)$，即排除当前词的主题分布。
这里的联合分布：
$$
p(\vec{w},\vec{z} | \vec{\alpha},\vec{\beta}) = p(\vec{w} | \vec{z},\vec{\beta})p(\vec{z} | \vec{\alpha})
$$
第一项因子是给定主题采样词的过程。后面的因此计算，$n_z^{(t)}$表示词$t$被观察到分配给主题$z$的次数，$n_m^{(k)}$表示主题$k$分配给文档$m$的次数。
$$
p(\vec{w} | ,\vec{z},\vec{\beta}) 
= \int p(\vec{w} | \vec{z},\vec{\Phi})p(\Phi | \vec{\beta})d \Phi \\
= \int \prod_{z=1}^{K} \frac{1}{\Delta(\vec{\beta})}\prod_{t=1}^V \phi_{z,t}^{n_z^{(t)}  + \beta_t - 1}d\vec{\phi_z} \\
= \prod_{z=1}^{K}\frac{\Delta(\vec{n_z} + \vec{\beta})}{\Delta(\vec{ \beta })} ,
\vec{n_z} = \{ n_z^{(t)} \}_{t=1}^V
$$
$$
p(\vec{z} | \vec{\alpha}) \\
 =  \int p(\vec{z} | \Theta) p(\Theta|\vec{\alpha}) d\Theta \\
= \int \prod_{m=1}^{M} \frac{1}{\Delta(\vec\alpha)} \prod_{k=1}^K\theta_{m,k}^{ n_m^{(k)} + \alpha_k - 1 }d\vec{\theta_m} \\
= \prod_{m=1}^M \frac{ \Delta(\vec{n_m} + \vec\alpha) }{ \Delta(\vec\alpha) }, \vec{n_m}=\{ n_m^{(k)} \}_{k=1}^K
$$
## Gibbs Updating Rule

![](http://7xkt0f.com1.z0.glb.clouddn.com/6173FC5A-A728-4818-9474-21D6E2A61CC2.png)

## 词分布和主题分布总结
经过上面的Gibbs采样，各个词所被分配到的主题已经完成了收敛，在这里就可以计算出文档属于主题的概率以及词属于文档的概率了。
$$
\phi_{k,t} = \frac{ n_k^{(t)} + \beta_t }{  \sum^V_{t=1}n_k^{(t)} + \beta_t } \\
\theta_{m,k} = \frac{ n_m^{(k)} + \alpha_k }{  \sum^K_{k=1}n_m^{(k)} + \alpha_k } \\
$$
$$
p(\vec{\theta_m} | \vec{z_m}, \vec{\alpha} ) 
= \frac{1}{Z_{\theta_m}} \prod_{n=1}^{N_m} p(z_{m,n} | \vec{\theta_m} * p(\vec{\theta_m} | \vec{alpha} )) 
= Dir(\vec{\theta_m} | \vec{n_m} + \vec{\alpha}) 
\\
p(\vec{\phi_k} | \vec{z}, \vec{w}, \vec{\beta} ) =
\frac{1}{Z_{\phi_k}} \prod_{i:z_i=k} p(w_i | \vec{\phi_k}) * p(\vec{\phi_k} | \vec{\beta})
= Dir(\vec{\phi_k} | \vec{n_k} + \vec{\beta})
$$

# 代码实现
代码的输入有文档数目$M$、词的数目$V$(非重复的term)、主题数目$K$，以及用$d$表示第几个文档，$k$表示主题，$w$表示词汇(term)，$n$表示词(word)。
$z[d][w]$:第$d$篇文档的第$w$个词来自哪个主题。$M$行，$X$列，$X$为对应的文档长度：即词(可重复)的数目。
$nw[w][t]$:第w个词是第t个主题的次数。word-topic矩阵，列向量$nw[][t]$表示主题t的词频数分布；V行K列。
$nd[d][t]$:第d篇文档中第t个主题出现的次数，doc-topic矩阵，行向量$nd[d]$表示文档$d$的主题频数分布。M行，K列。
辅助向量：
$ntSum[t]$:第t个主题在所有语料出现的次数，K维
$ndSum[d]$:第d篇文档中词的数目(可重复)，M维
$P[t]$:对于当前计算的某词属于主题t的概率，K维




# 超参数的确定
- 交叉验证
- $\alpha$表达了不同文档间主题是否鲜明，$\beta$度量了有多少近义词能够属于同一个类别。
- 给定主题数目$K$，可以使用：
$$
\alpha = 50 / K \\
\beta = 0.01

$$
