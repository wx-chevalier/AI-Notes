# Colaboratory

Colaboratory 是一个免费的 Jupyter 笔记本环境，不需要进行任何设置就可以使用，并且完全在云端运行。借助 Colaboratory，我们可以在浏览器中编写和执行代码、保存和共享分析结果，以及利用强大的计算资源。

Colab 能够方便地与 Google Driver 与 Github 链接，我们可以使用 [Open in Colab](https://chrome.google.com/webstore/detail/open-in-colab/iogfkhleblhcpcekbiedikdehleodpjo) 插件快速打开 Github 上的 Notebook，或者使用类似于 https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb 这样的链接打开。如果需要将 Notebook 保存回 Github，直接使用 `File→Save a copy to GitHub` 即可。

# 依赖处理

Colab 提供了便捷的依赖安装功能，允许使用 pip 或者 apt-get 命令进行安装：

```sh
# Importing a library that is not in Colaboratory
!pip install -q matplotlib-venn
!apt-get -qq install -y libfluidsynth1

# Upgrading TensorFlow
# To determine which version you're using:
!pip show tensorflow

# For the current version:
!pip install --upgrade tensorflow

# For a specific version:
!pip install tensorflow==1.2

# For the latest nightly build:
!pip install tf-nightly

# Install 7zip reader libarchive
# https://pypi.python.org/pypi/libarchive
!apt-get -qq install -y libarchive-dev && pip install -q -U libarchive
import libarchive

# Install GraphViz & PyDot
# https://pypi.python.org/pypi/pydot
!apt-get -qq install -y graphviz && pip install -q pydot
import pydot

# Install cartopy
!apt-get -qq install python-cartopy python3-cartopy
import cartopy
```

# 数据加载

# 控件使用

## 网格

## 表单

值得称道的是，Colab 还提供了便捷的

# 硬件加速

我们可以通过如下方式查看 Colab 为我们提供的硬件：

```py
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

!ls /proc
# CPU信息
!cat /proc/cpuinfo
# 内存
!cat /proc/meminfo
# 版本
!cat /proc/version
# 设备
!cat /proc/devices
# 空间
!df
```

如果需要为 Notebook 启动 GPU 支持：`Click Edit->notebook settings->hardware accelerator->GPU`，然后在代码中判断是否有可用的 GPU 设备：

```py
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
```

我们可以通过构建经典的 CNN 卷积层来比较 GPU 与 CPU 在运算上的差异：

```py
import tensorflow as tf
import timeit

# See https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.device('/cpu:0'):
  random_image_cpu = tf.random_normal((100, 100, 100, 3))
  net_cpu = tf.layers.conv2d(random_image_cpu, 32, 7)
  net_cpu = tf.reduce_sum(net_cpu)

with tf.device('/gpu:0'):
  random_image_gpu = tf.random_normal((100, 100, 100, 3))
  net_gpu = tf.layers.conv2d(random_image_gpu, 32, 7)
  net_gpu = tf.reduce_sum(net_gpu)

sess = tf.Session(config=config)

# Test execution once to detect errors early.
try:
  sess.run(tf.global_variables_initializer())
except tf.errors.InvalidArgumentError:
  print(
      '\n\nThis error most likely means that this notebook is not '
      'configured to use a GPU.  Change this in Notebook Settings via the '
      'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
  raise

def cpu():
  sess.run(net_cpu)

def gpu():
  sess.run(net_gpu)

# Runs the op several times.
print('Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images '
      '(batch x height x width x channel). Sum of ten runs.')
print('CPU (s):')
cpu_time = timeit.timeit('cpu()', number=10, setup="from __main__ import cpu")
print(cpu_time)
print('GPU (s):')
gpu_time = timeit.timeit('gpu()', number=10, setup="from __main__ import gpu")
print(gpu_time)
print('GPU speedup over CPU: {}x'.format(int(cpu_time/gpu_time)))

sess.close()
```
