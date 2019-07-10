# Titanic

这里我们以 [Scikit Flow](https://github.com/google/skflow)为例，scflow 是 Google 官方提供的基于 scikit api 的对于 TensorFlow 的封装，整个开发环境安装如下：

```python
pip install numpy scipy sklearn pandas
# For Ubuntu:
pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl
# For Mac:
pip install https://storage.googleapis.com/tensorflow/mac/tensorflow-0.5.0-py2-none-any.whl
pip install git+git://github.com/google/skflow.git
```

完整的数据集合和代码在这里：[http://github.com/ilblackdragon/tf\_examples](http://github.com/ilblackdragon/tf_examples)

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

