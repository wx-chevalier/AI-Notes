

# Tensorflow

## Reference

# QuickStart

> [tensorflow-googles-latest-machine](http://googleresearch.blogspot.com/2015/11/tensorflow-googles-latest-machine_9.html)

## Installation

![](http://1.bp.blogspot.com/-vDKYuCD8Gyg/Vj0B3BEQfXI/AAAAAAAAAyA/9tWmYUOxo0g/s1600/cifar10_2.gif)

因为众所周知的原因，在国内搭建 Tensorflow 的环境又经历了一些波折。笔者习惯用 Docker 作为复杂依赖项目的开发环境，Google 提供的安装方式有如下几个。

### Binary Installation

TensorFlow 的 Python 的 API 是 2.7，最简单的方式就是在 MAC 或者 Unix 上使用 pip 命令导入。

* Linux

```
# For CPU-only version
$ pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl

# For GPU-enabled version (only install this version if you have the CUDA sdk installed)
$ pip install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl
```

* Mac OS X

```
# Only CPU-version is available at the moment.
$ pip install https://storage.googleapis.com/tensorflow/mac/tensorflow-0.5.0-py2-none-any.whl
```

### Docker

笔者就是用的这个方式，不过可能比较笨吧，之前用的代理下载 DockerHub 的镜像都没问题，但是死活下不了 TensorFlow，只能默默开了个 VPS 下了镜像然后打包成 tar 再拖到本地导入，笔者打包的 tar 文件在[这里](http://7xlgth.com1.z0.glb.clouddn.com/DQ9S2MI7jAnx2HjotLLxSTObTBYyjCYrEhBQNwVF)。导入到本地的 Docker 镜像库中只需要：

```shell
docker load < /tmp/tensorflow.tar
```

具体的 Docker 的安装过程可以参考笔者的其他文章，镜像下载导入好了之后直接：

```
docker run -it b.gcr.io/tensorflow/tensorflow
```

这个默认的镜像比较小，只包含了一些必要的运行条件，Google 还提供了一个更完整的镜像`b.gcr.io/tensorflow/tensorflow-full`。

如果需要重新编译的话：

* tensorflow/tensorflow

```
$ docker build -t $USER/tensorflow -f Dockerfile.lite .
```

* tensorflow/tensorflow-full

同样需要依赖于 tensor flow 的基础镜像，所以还是得翻墙

```
$ git clone https://github.com/tensorflow/tensorflow
$ docker build -t $USER/tensorflow-full -f Dockerfile.cpu .
```

* [](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/docker#tensorflowtensorflow-gpu)tensorflow/tensorflow-gpu

这个需要的稍微复杂一点：

```
$ cp -a /usr/local/cuda .
$ docker build -t $USER/tensorflow-gpu-base -f Dockerfile.gpu_base .
# Flatten the image
$ export TC=$(docker create $USER/tensorflow-gpu-base)
$ docker export $TC | docker import - $USER/tensorflow-gpu-flat
$ docker rm $TC
$ export TC=$(docker create $USER/tensorflow-gpu-flat /bin/bash)
$ docker commit --change='CMD ["/bin/bash"]'  --change='ENV CUDA_PATH /usr/local/cuda' --change='ENV LD_LIBRARY_PATH /usr/local/cuda/lib64' --change='WORKDIR /root' $TC $USER/tensorflow-full-gpu
$ docker rm $TC
```

### VirtualEnv

这也是 Google 官方推荐的一种构建方式，virtualenv 是 Python 领域的一种环境管理，首先是安装所有必备工具：

```
# On Linux:
$ sudo apt-get install python-pip python-dev python-virtualenv

# On Mac:
$ sudo easy_install pip  # If pip is not already installed
$ sudo pip install --upgrade virtualenv
```

其次是创建一个新的工作区间：

```
$ virtualenv --system-site-packages ~/tensorflow
$ cd ~/tensorflow
```

接下来是启用这个工作区间：

```
$ source bin/activate  # If using bash
$ source bin/activate.csh  # If using csh
(tensorflow)$  # Your prompt should change
```

然后在该工作区间中，安装 TensorFlow：

```
(tensorflow)$ pip install --upgrade <$url_to_binary.whl>
```

最后直接运行程序：

```
(tensorflow)$ python tensorflow/models/image/mnist/convolutional.py

# When you are done using TensorFlow:
(tensorflow)$ deactivate  # Deactivate the virtualenv

$  # Your prompt should change back
```

## Example

有人在 Github 上开源了一波 TensorFlow 的示范教程，大概是这里[TensorFlow-Tutorials](https://github.com/nlintz/TensorFlow-Tutorials)。

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
