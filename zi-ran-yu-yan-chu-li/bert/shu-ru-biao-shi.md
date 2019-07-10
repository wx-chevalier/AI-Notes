# 输入表示

## 输入表示

Input: 每个输入序列的第一个 token \[CLS\]专门用来分类, 直接利用此位置的最后输出作为分类任务的输入 embedding。

![](https://i.postimg.cc/5yKzdG1s/image.png)

从直观上来说，在预训练时，\[CLS\]不参与 mask，因而该位置面向整个序列的所有 position 做 attention，\[CLS\]位置的输出足够表达整个句子的信息，类似于一个 global feature；而单词 token 对应的 embedding 更关注该 token 的语义语法及上下文信息表达，类似于一个 local feature。

Position Embeddings: transformer 的 Position Encoding 是通过 sin, cos 直接构造出来的，Position Embeddings 是通过模型学习到的 embedding 向量，最高支持 512 维。

Segment Embeddings：在预训练的句对预测任务及问答、相似匹配等任务中，需要对前后句子做区分，将句对输入同一序列，以特殊标记\[SEP\]分割，同时对第一个句子的每个 token 添加 Sentence A Embedding, 第二个句子添加 Sentence B Embedding,实验中让 $E\_A$ =1, $E\_B$ =0。

## Fine-tune

针对不同任务，BERT 采用不同部分的输出做预测,分类任务利用 \[CLS\] 位置的 embedding，NER 任务利用每个 token 的输出 embedding。

![](https://i.postimg.cc/kXD8FnFz/image.png)

