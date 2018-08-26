# TensorFlow

Google 最近开源了机器学习框架 TensorFlow，在很短的时间内就在 Github 上获得了超过的 10K 的赞，并且在 AI 研究者之间引发了很大的反响。

# Why do I care?

在了解 TensorFlow 之前，我们首先要搞明白一个问题。作为一个专业的数据科学家，为什么在有了大量现存的数据科学或者机器学习的工具(譬如 R,SciKit Learn)之后，还需要关注其他的机器学习框架，笔者窃以为有以下两点：

- TensorFlow 中的深度学习部分允许使用者将多个不同的模型或者转化结合到一个模型中，并且同时训练它们。根据 TensorFlow 设定的不同的 OP，你可以同时处理文本、图片和其他的常规的类别或者连续变量。开发者可以方便地同时进行多目标或者多损失函数的训练，而其他很多的机器学习框架并不能在传统的模型建立时候做到这一点。
- TensorFlow 中的管道处理方式会成为数据处理的很重要的一个角色。未来，数据处理与机器学习将会在一个框架中同时进行，而 TensorFlow 正是在向这个方向前行。

# 基于 Titanic 数据集的简单模型

这里我们以 [Scikit Flow](https://github.com/google/skflow)为例，scflow 是 Google 官方提供的基于 scikit api 的对于 TensorFlow 的封装，整个开发环境安装如下：

```
pip install numpy scipy sklearn pandas
# For Ubuntu:
pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl
# For Mac:
pip install https://storage.googleapis.com/tensorflow/mac/tensorflow-0.5.0-py2-none-any.whl
pip install git+git://github.com/google/skflow.git
```

完整的数据集合和代码在这里：[http://github.com/ilblackdragon/tf_examples](http://github.com/ilblackdragon/tf_examples)

首先我们来看下数据的格式：

```python
>>> import pandas
>>> data = pandas.read_csv('data/train.csv')
>>> data.shape
(891, 12)
>>> data.columns
Index([u'PassengerId', u'Survived', u'Pclass', u'Name', u'Sex', u'Age',
       u'SibSp', u'Parch', u'Ticket', u'Fare', u'Cabin', u'Embarked'],
      dtype='object')
>>> data[:1]
   PassengerId  Survived  Pclass                     Name   Sex  Age  SibSp
0            1         0       3  Braund, Mr. Owen Harris  male   22      1
   Parch     Ticket  Fare Cabin Embarked
0      0  A/5 21171  7.25   NaN        S
```

下面我们首先用 Scikit 中提供的 LogisticRegression 来判断下 Survived 的类别：

```python
>>> y, X = train['Survived'], train[['Age', 'SibSp', 'Fare']].fillna(0)
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
>>> lr = LogisticRegression()
>>> lr.fit(X_train, y_train)
>>> print accuracy_score(lr.predict(X_test), y_test)
0.664804469274
```

在代码中我们将数据集分为了特征与目标两个属性，并且将所有的 N/A 数据设置为了 0，并建立了一个 Logistic 回归。并且最后对该模型的准确率进行了一个检测。接下来，我们尝试使用 Scikit Flow 的类似的接口：

```python
>>> import skflow
>>> import random
>>> random.seed(42) # to sample data the same way
>>> classifier = skflow.TensorFlowLinearClassifier(n_classes=2, batch_size=128, steps=500, learning_rate=0.05)
>>> classifier.fit(X_train, y_train)
>>> print accuracy_score(classifier.predict(X_test), y_test)
0.68156424581
```

# Scikit Flow

Scikit Flow 封装了很多的 TensorFlow 的最新的 API，并且将它们封装成了很类似于 Scikit Learn API 的样式。TensorFlow 的核心即是基于构建与执行某个图，这是一个非常棒，但也是非常难以直接上手的概念。如果我们看 Scikit Flow 的底层封装，我们可以看到整个模型被分为了以下几个部分：

- **TensorFlowTrainer** —  用于寻找所有优化器的类(使用梯度进行了部分的图构建，进行了一些梯度裁剪并且添加一些优化器)
- **logistic_regression** —用于构造 Logistic 回归图的函数
- **linear_regression** —  用于构造线性回归图的函数
- **DataFeeder** —  用于将训练数据填充到模型中 (由于 TensorFlow 使用了数据集合中的随机的一些部分作为随机梯度下降的数据，因此需要这样的 Mini 数据批处理)。
- **TensorFlowLinearClassifier** —  用 LogisticRegression 模型实现了 Scikit Learn 提供的某个接口。它提供了一个模型和一个训练器，并且根据给定的数据集合利用 fit()方法进行数据训练，并且通过 predict()方法进行预测。
- **TensorFlowLinearRegressor** —  类似于 TensorFlowClassifier, 但是使用 LinearRegression 作为模型。

如果你本身对于 TensorFlow 就已经很熟悉了，那么 Scikit Flow 会更加的易于上手。

完整的代码列举如下：

```python
import random
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import check_array
from sklearn.cross_validation import train_test_split

import tensorflow as tf

import skflow

train = pandas.read_csv('data/titanic_train.csv')
y, X = train['Survived'], train[['Age', 'SibSp', 'Fare']].fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression()
lr.fit(X_train, y_train)
print accuracy_score(lr.predict(X_test), y_test)


# Linear classifier.

random.seed(42)
tflr = skflow.TensorFlowLinearClassifier(n_classes=2, batch_size=128,
                                         steps=500, learning_rate=0.05)
tflr.fit(X_train, y_train)
print accuracy_score(tflr.predict(X_test), y_test)

# 3 layer neural network with rectified linear activation.

random.seed(42)
classifier = skflow.TensorFlowDNNClassifier(hidden_units=[10, 20, 10],
    n_classes=2, batch_size=128, steps=500, learning_rate=0.05)
classifier.fit(X_train, y_train)
print accuracy_score(classifier.predict(X_test), y_test)

# 3 layer neural network with hyperbolic tangent activation.

def dnn_tanh(X, y):
    layers = skflow.ops.dnn(X, [10, 20, 10], tf.tanh)
    return skflow.models.logistic_regression(layers, y)

random.seed(42)
classifier = skflow.TensorFlowEstimator(model_fn=dnn_tanh,
    n_classes=2, batch_size=128, steps=500, learning_rate=0.05)
classifier.fit(X_train, y_train)
print accuracy_score(classifier.predict(X_test), y_test)
```

原文地址：[这里](https://medium.com/@ilblackdragon/tensorflow-tutorial-part-2-9ffe47049c92#.1yipdfg3k)

本部分我们将继续深入并且尝试构建多层全连接的神经网络，并且自定义网络模型并在此基础上尝试卷积网络。

# Multi-layer fully connected neural network

当然，这里并没有太多关于其他的线性/Logistic 拟合的框架。TensorFlow 一个基础的理念就是希望能够将模型的不同的部分连接并且使用相关的代价函数去进行参数优化。Scikit Flow 已经提供了非常便捷的封装以供创建多层全连接单元，因此只需要简单地将分类器替换为 TensorFlowDNNClassifier 然后指定它的各个参数，就可以进行相应的训练与预测。

```python
>>> classifier = skflow.TensorFlowDNNClassifier(
...     hidden_units=[10, 20, 10],
...     n_classes=2,
...     batch_size=128,
...     steps=500,
...     learning_rate=0.05)
>>> classifier.fit(X_train, y_train)
>>> score = accuracy_score(classifier.predict(X_test), y_test)
>>> print("Accuracy: %f" % score)
Accuracy: 0.67597765363
```

上述程序会用 10,20 以及 10 个独立的隐藏单元创建一个 3 层的全连接网络，并且使用默认的 Rectified 激活函数。关于这个激活函数的自定义将会在下面讲到。

> 模型中的参数有一个示例，但是在实际的应用中，学习速率、优化器以及训练步长的不同可能会导致结果有很大的差异性。一般情况下，我们会使用类似于超参数搜索的方法来寻找一个最优的组合。

# Multi-layer with tanh activation

笔者并没有进行太多的参数搜索，但是之前的 DNN 模型确实抛出了一个比 Logistic 回归还要差的结果。可能这是因为过拟合或者欠拟合的情形。

为了解决这个问题，笔者打算将上文中用的 DNN 模型转化为自定义的模型：

```python
>>> def dnn_tanh(X, y):
...    layers = skflow.ops.dnn(X, [10, 20, 10], tf.tanh)
...    return skflow.ops.logistic_classifier(layers, y)

>>> classifier = skflow.TensorFlowEstimator(
...     model_fn=dnn_tanh,
...     n_classes=2,
...     batch_size=128,
...     steps=500,
...     learning_rate=0.05)
>>> classifier.fit(X_train, y_train)
>>> score = accuracy_score(classifier.predict(X_test), y_test)
>>> print("Accuracy: %f" % score)
Accuracy: 0.692737430168
```

这个模型很类似之前那个，但是我们将激活方程从线性整流变成了双曲正切。正如你所见，创建一个自定义的模型还是很简答的，就是输入 X 与 y 这两个 Tensors，然后返回 prediction 与 loss 这两个 Tensor。

# Digit recognition

TensorFlow 的教程当中自然应该包含一波数字识别的测试：

```python
import random
from sklearn import datasets, cross_validation, metrics
import tensorflow as tf

import skflow

random.seed(42)

# Load dataset and split it into train / test subsets.

digits = datasets.load_digits()
X = digits.images
y = digits.target

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
    test_size=0.2, random_state=42)

# TensorFlow model using Scikit Flow ops

def conv_model(X, y):
    X = tf.expand_dims(X, 3)
    features = tf.reduce_max(skflow.ops.conv2d(X, 12, [3, 3]), [1, 2])
    features = tf.reshape(features, [-1, 12])
    return skflow.models.logistic_regression(features, y)

# Create a classifier, train and predict.
classifier = skflow.TensorFlowEstimator(model_fn=conv_model, n_classes=10,
                                        steps=500, learning_rate=0.05,
                                        batch_size=128)
classifier.fit(X_train, y_train)
score = metrics.accuracy_score(classifier.predict(X_test), y_test)
print('Accuracy: %f' % score)
```

我们自定义了 `con_model` 函数，使用 Tensor X 以及 y 作为参数，使用最大化池来创建一个二维的卷积层。这个层的结果作为参数传给了 logistic 拟合，在其中将会来处理具体的分类问题。我们只需要按照自身的需求来添加不同的层即可以完成一些复杂的图片识别或者其他处理操作。

### Multiply

```python
import tensorflow as tf

a = tf.placeholder("float") # Create a symbolic variable 'a'
b = tf.placeholder("float") # Create a symbolic variable 'b'

y = tf.mul(a, b) # multiply the symbolic variables

sess = tf.Session() # create a session to evaluate the symbolic expressions

print "%f should equal 2.0" % sess.run(y, feed_dict={a: 1, b: 2}) # eval expressions with parameters for a and b
print "%f should equal 9.0" % sess.run(y, feed_dict={a: 3, b: 3})
```

### Linear Regression

```python
import tensorflow as tf
import numpy as np

trX = np.linspace(-1, 1, 101)
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33 # create a y value which is approximately linear but with some random noise

X = tf.placeholder("float") # create symbolic variables
Y = tf.placeholder("float")


def model(X, w):
    return tf.mul(X, w) # lr is just X*w so this model line is pretty simple


w = tf.Variable(0.0, name="weights") # create a shared variable (like theano.shared) for the weight matrix
y_model = model(X, w)

cost = (tf.pow(Y-y_model, 2)) # use sqr error for cost function

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost) # construct an optimizer to minimize cost and fit line to my data

sess = tf.Session()
init = tf.initialize_all_variables() # you need to initialize variables (in this case just variable W)
sess.run(init)

for i in range(100):
    for (x, y) in zip(trX, trY):
        sess.run(train_op, feed_dict={X: x, Y: y})

print(sess.run(w))  # something around 2
```

### Logistic Regression

```python
import tensorflow as tf
import numpy as np
import input_data


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w):
    return tf.matmul(X, w) # notice we use the same model as linear regression, this is because there is a baked in cost function which performs softmax and cross entropy


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder("float", [None, 784]) # create symbolic variables
Y = tf.placeholder("float", [None, 10])

w = init_weights([784, 10]) # like in linear regression, we need a shared variable weight matrix for logistic regression

py_x = model(X, w)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y)) # compute mean cross entropy (softmax is applied internally)
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct optimizer
predict_op = tf.argmax(py_x, 1) # at predict time, evaluate the argmax of the logistic regression

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(100):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
    print i, np.mean(np.argmax(teY, axis=1) ==
                     sess.run(predict_op, feed_dict={X: teX, Y: teY}))
```
