# Conda

笔者推荐使用 Anaconda 作为环境搭建工具，并且推荐使用 Python 3.5 版本，可以在[这里](https: // www.continuum.io/downloads)下载。如果是习惯使用 Docker 的小伙伴可以参考[anaconda-notebook](https: // github.com/rothnic/anaconda-notebook)

```sh
docker pull rothnic/anaconda-notebook
docker run - p 8888: 8888 - i - t rothnic/anaconda-notebook
```

安装完毕之后可以使用如下命令验证安装是否完毕:

```sh
conda - -version
```

安装完毕之后我们就可以创建具体的开发环境了，主要是通过 create 命令来创建新的独立环境:

```
conda create - -name snowflakes biopython
```

该命令会创建一个名为 snowflakes 并且安装了 Biopython 的环境，如果你需要切换到该开发环境，可以使用 activate 命令:

- Linux, OS X: `source activate snowflakes`
- Windows: `activate snowflakes`

我们也可以在创建环境的时候指明是用 python2 还是 python3:

```
conda create - -name bunnies python = 3 astroid babel
```

环境创建完毕之后，我们可以使用`info`命令查看所有环境:

```
conda info - -envs
conda environments:


          snowflakes          * /home/username/miniconda/envs/snowflakes
          bunnies               / home/username/miniconda/envs/bunnies
          root                  / home/username/miniconda
```

当我们切换到某个具体的环境后，可以安装依赖包了:

```
conda list  # 列举当前环境中的所有依赖包
conda install nltk  # 安装某个新的依赖
```
