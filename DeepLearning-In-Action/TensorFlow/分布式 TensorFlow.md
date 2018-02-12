
# Distributed TensorFlow

本目录包括了运行时分布式TensorFlow的实现，其底层使用了[gRPC](http://grpc.io) 作为进程内通信的支持库。

## Quick start

首先，需要构建一个TensorFlow的服务端可执行版本(`grpc_tensorflow_server`) 以及一个基于gRPC的客户端。目前只能基于源代码进行自构建, 但是会包含在未来发布的二进制版本中。可以使用如下命令进行构建:

```shell
# CPU-only build.
$ bazel build -c opt //tensorflow/core/distributed_runtime/rpc:grpc_tensorflow_server

# GPU build.
$ bazel build -c opt --config=cuda //tensorflow/core/distributed_runtime/rpc:grpc_tensorflow_server
```

如果是从最新的源代码创建的Python依赖包，它会自动包含一个基于gRPC的客户端。如果使用的是一个之前发布的二进制版本，需要根据这个[安装说明](https://www.tensorflow.org/versions/master/get_started/os_setup.html#create-the-pip-package-and-install)来重新编译安装。在你成功地构建了分布式的TensorFlow组件之后，可以通过如下方式来启动服务器并且判断你的安装是否成功：

```shell
# Start a TensorFlow server as a single-process "cluster".
$ bazel-bin/tensorflow/core/distributed_runtime/rpc/grpc_tensorflow_server \
    --cluster_spec='local|localhost:2222' --job_name=local --task_index=0 &
```

然后启动Python的交互器并且启动一个Session：

```python
$ python
>>> import tensorflow as tf
>>> c = tf.constant("Hello, distributed TensorFlow!")
>>> sess = tf.Session("grpc://localhost:2222")
>>> sess.run(c)
'Hello, distributed TensorFlow!'
```

## 集群定义

命令行参数 `grpc_tensorflow_server` 定义了集群之间的关系. 参数 `--cluster_spec` 决定了集群中工作对象的多少, 譬如有一系列的 *jobs*, 而每个*jobs*又包含了多个*task* 终端。 所有集群中的处理过程都必须设置相同的 `--cluster_spec`参数， 例子如下:

| `--cluster_spec='...'`                   | Available tasks                          |
| ---------------------------------------- | ---------------------------------------- |
| `local\|localhost:2222`                  | `/job:local/task:0`                      |
| `local\|localhost:2222;localhost:2223`   | `/job:local/task:0``/job:local/task:1`   |
| `worker\|worker0:2222;worker1:2222;worker2:2222,``ps\|ps0:2222;ps1:2222` | `/job:worker/task:0``/job:worker/task:1``/job:worker/task:2``/job:ps/task:0``/job:ps/task:1` |

还有 `--job_name` 与 `--task_index` 标志位指明了哪些任务会运行在当前处理过程上。 具体而言,
`--job_name=local --task_index=0` 意思就是该过程会被标志为
`/job:local/task:0`, 然后所有在该过程上的TensorFlow的设备都会使用这个前缀。

**N.B.** 
手动来指明这些运行参数可能是非常冗长的，特别是对一个大型集群而言。我们正在研发可以程式化启动的工具，譬如使用一些类似于[Kubernetes](http://kubernetes.io)集群管理器。如果有啥集群管理工具你觉得挺好的希望加入进来，可以在[GitHub issue](https://github.com/tensorflow/tensorflow/issues)上提出你的建议。

## 标注模型中的分布式设备

为了将某个操作放在某个特殊的处理过程上,在分布式环境下依然可以使用
[`tf.device()`](https://www.tensorflow.org/versions/master/api_docs/python/framework.html#device)
函数，之前是用来指明是放在CPU还是GPU上的。譬如:

```python
with tf.device("/job:ps/task:0"):
  weights_1 = tf.Variable(...)
  biases_1 = tf.Variable(...)
  
with tf.device("/job:ps/task:1"):
  weights_2 = tf.Variable(...)
  biases_2 = tf.Variable(...)
  
with tf.device("/job:worker/task:7"):
  input, labels = ...
  layer_1 = tf.nn.relu(tf.matmul(input, weights_1) + biases_1)
  logits = tf.nn.relu(tf.matmul(layer_1, weights_2) + biases_2)
  # ...
  train_op = ...

with tf.Session("grpc://worker7:2222") as sess:
  for _ in range(10000):
    sess.run(train_op)
```
在上面的例子中，Variables在job `ps`的两个task上被创建，然后计算密集型的部分创建在job `work`上。TensorFlow会自动地在不同的job之间传输数据。（从`job`到`work`是前向传递，而从`worker`到`ps`是梯度应用）。


## Replicated Computation
一个常见的训练配置(数据并行训练)包含了job `ps`上共享参数以及job `work`上的多个任务来训练相同的模型。每个task一般会运行在不同的机器上。现在还是有很多办法可以在TensorFlow中来实现这一种结构的，我们未来也会提供更简单的实现方式，主要途径有：

* 构建单一的包含了一系列参数的图(in `tf.Variable` nodes pinned to `/job:ps`), 并且创建多个模型的副本来映射到`/job:worker`中的不同tasks。每个model的副本有一个不同的`train_op`，并且对于每个worker `i`而言一个或者多个的客户端线程可以调用`sess.run(train_ops[i])`。这种方法使用了单一的`tf.Session`，它的工作目标是集群中的某个workers。
  
  
* As above, but where the gradients from all workers are averaged. See the
  [CIFAR-10 multi-GPU trainer](https://www.tensorflow.org/code/tensorflow/models/image/cifar10/cifar10_multi_gpu_train.py)
  for an example of this form of replication. The implements *synchronous* training
  
* 另一种分布式训练器的方法使用多张图，一张图对应一个worker，并且每张图都包含了一系列的参数的集合(`/job:ps`)和一份模型的赋值。而容器的机制就是在不同的图之间共享变量：一旦某个变量构造完成，可选的`container`参数会由图中每份复制的相同值来决定。对于较大的模型而言，这种方法会更加有效，毕竟整个图更小了一点。
这种方法使用多个`tf.Session`对象：每个worker过程都会包含一个，不过不同的Session会指向不同的目标worker。这个`tf.Session`对象即可以在单一的Python客户端中创建，也可以在多个客户端中创建。

## 术语

**Client**
一个典型的客户端一般会构建一个TensorFlow的图并且使用`tensorflow::Session`来完成与集群的交互。客户端一般会用Python或者C++编写，一般来说一个客户端可以同时与多个服务端进行交互（参考上文的重复训练），并且一个服务端也可以同时服务于多个客户端。

**Cluster**
一个TensorFlow集群会包含一个或者多个TensorFlow的服务端，被切分为一系列命名的job，而每个job又会负责一系列的tasks。一个集群一般会专注于一个相对高层的目标，譬如用多台机器并行地训练一个神经网络。

**Job**
一个job会包含一系列的致力于某个相同目标的task。譬如，一个叫`ps`（意思是参数服务）的job会用于处理存储于更新Variables相关的工作。而一个叫`worker`的job会用于承载那些用于计算密集型的无状态节点。一般来说一个job中的tasks会运行在不同的机器中。


**Master service**
Master Service是一个RPC服务用于与一系列远端的分布式设备进行交互。Master Service实现了`tensorflow::Session` 接口, 并且用来协调多个worker service。

**Task**
一个Task一般会关联到某个单一的TensorFlow服务端的处理过程，属于一个特定的job并且在该job的任务列表中有个唯一的索引。

**TensorFlow server**
用于运行grpc_tensorflow_server的处理过程，是一个集群中的一员，并且想外暴露了一个Master Service与一个Worker Service。

**Worker service**
一个执行部分TensorFlow图部分内容的RPC服务。

