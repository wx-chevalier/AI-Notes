> 本文从属于笔者的[爬虫与搜索引擎 最佳实践](https://github.com/wxyyxc1992/datascience-practice-handbook/blob/master/CrawlerSE/README.md)系列文章

# Introduction

ElasticSearch是一个基于[Apache Lucene(TM)](https://lucene.apache.org/core/)的开源搜索引擎。无论在开源还是专有领域，Lucene可以被认为是迄今为止最先进、性能最好的、功能最全的搜索引擎库。但是，Lucene只是一个库。想要使用它，你必须使用Java来作为开发语言并将其直接集成到你的应用中，更糟糕的是，Lucene非常复杂，你需要深入了解检索的相关知识来理解它是如何工作的。ElasticSearch也使用Java开发并使用Lucene作为其核心来实现所有索引和搜索的功能，但是它的目的是通过简单的`RESTful API`来隐藏Lucene的复杂性，从而让全文搜索变得简单。
不过，Elasticsearch不仅仅是Lucene和全文搜索，我们还能这样去描述它：
- 分布式的实时文件存储，每个字段都被索引并可被搜索
- 分布式的实时分析搜索引擎
- 可以扩展到上百台服务器，处理PB级结构化或非结构化数据

而且，所有的这些功能被集成到一个服务里面，你的应用可以通过简单的`RESTful API`、各种语言的客户端甚至命令行与之交互。上手Elasticsearch非常容易。它提供了许多合理的缺省值，并对初学者隐藏了复杂的搜索引擎理论。它开箱即用（安装即可使用），只需很少的学习既可在生产环境中使用。在ElasticSearch中，我们常常会听到Index、Type以及Document等概念，那么它们与传统的熟知的关系型数据库中名称的类比如下：
```
Relational DB -> Databases -> Tables -> Rows -> Columns
Elasticsearch -> Indices   -> Types  -> Documents -> Fields
```

这里借用[此文](http://www.cnblogs.com/xing901022/p/4704319.html)的一张思维脑图来描述整个ElasticSearch生态圈你所应该了解的内容:
![](http://images0.cnblogs.com/blog2015/449064/201508/051805578611052.png)


## Reference

## Books & Tutorial

- [ElasticSearch权威指南中文版](http://es.xiaoleilu.com/010_Intro/00_README.html)
- [elasticsearch-definitive-guide](https://github.com/elastic/elasticsearch-definitive-guide)

# Quick Start

## Installation
在[这里](https://www.elastic.co/downloads/elasticsearch)下载ElasticSearch的最新预编译版本，然后直接解压缩启动即可。笔者此时使用的是2.3.5版本的ElasticSearch，其文件目录结构如下：
```
home---这是Elasticsearch解压的目录
　　bin---这里面是ES启动的脚本

　　conf---elasticsearch.yml为ES的配置文件

　　data---这里是ES得当前节点的分片的数据，可以直接拷贝到其他的节点进行使用

　　logs---日志文件

　　plugins---这里存放一些常用的插件，如果有一切额外的插件，可以放在这里使用。
```
在ElasticSearch 2.x版本中，默认是不允许以Root用户身份运行实例，可以使用`bin/elasticsearch -Des.insecure.allow.root=true`来以Root身份启动集群，此外还可以使用`bin/elasticsearch -f -Des.path.conf=/path/to/config/dir`参数来读取相关的`.yml`或者`.json`配置。

![](http://joelabrahamsson.com/PageFiles/669/0_running_elasticsearch_on_osx.png)

还有些常见的配置如下所示：
| Setting                        | Description                              |
| ------------------------------ | ---------------------------------------- |
| `http.port`                    | A bind port range. Defaults to `9200-9300`. |
| `http.publish_port`            | The port that HTTP clients should use when communicating with this node. Useful when a cluster node is behind a proxy or firewall and the `http.port` is not directly addressable from the outside. Defaults to the actual port assigned via `http.port`. |
| `http.bind_host`               | The host address to bind the HTTP service to. Defaults to `http.host`(if set) or `network.bind_host`. |
| `http.publish_host`            | The host address to publish for HTTP clients to connect to. Defaults to `http.host` (if set) or `network.publish_host`. |
| `http.host`                    | Used to set the `http.bind_host` and the `http.publish_host` Defaults to `http.host` or `network.host`. |
| `http.max_content_length`      | The max content of an HTTP request. Defaults to `100mb`. If set to greater than `Integer.MAX_VALUE`, it will be reset to 100mb. |
| `http.max_initial_line_length` | The max length of an HTTP URL. Defaults to `4kb` |
| `http.max_header_size`         | The max size of allowed headers. Defaults to `8kB` |
| `http.compression`             | Support for compression when possible (with Accept-Encoding). Defaults to `false`. |
| `http.compression_level`       | Defines the compression level to use. Defaults to `6`. |
| `http.cors.enabled`            | Enable or disable cross-origin resource sharing, i.e. whether a browser on another origin can do requests to Elasticsearch. Defaults to `false`. |
| `http.cors.allow-origin`       | Which origins to allow. Defaults to no origins allowed. If you prepend and append a `/` to the value, this will be treated as a regular expression, allowing you to support HTTP and HTTPs. for example using `/https?:\/\/localhost(:[0-9]+)?/` would return the request header appropriately in both cases. `*` is a valid value but is considered a **security risk** as your elasticsearch instance is open to cross origin requests from **anywhere**. |
| `http.cors.max-age`            | Browsers send a "preflight" OPTIONS-request to determine CORS settings. `max-age` defines how long the result should be cached for. Defaults to `1728000` (20 days) |
| `http.cors.allow-methods`      | Which methods to allow. Defaults to `OPTIONS, HEAD, GET, POST, PUT, DELETE`. |
| `http.cors.allow-headers`      | Which headers to allow. Defaults to `X-Requested-With, Content-Type, Content-Length`. |
| `http.cors.allow-credentials`  | Whether the `Access-Control-Allow-Credentials` header should be returned. Note: This header is only returned, when the setting is set to `true`. Defaults to `false` |
| `http.detailed_errors.enabled` | Enables or disables the output of detailed error messages and stack traces in response output. Note: When set to `false` and the`error_trace` request parameter is specified, an error will be returned; when `error_trace` is not specified, a simple message will be returned. Defaults to `true` |
| `http.pipelining`              | Enable or disable HTTP pipelining, defaults to `true`. |
| `http.pipelining.max_events`   | The maximum number of events to be queued up in memory before a HTTP connection is closed, defaults to `10000`. |

### REST API
在我们启动了某个ElasticSearch实例之后，即可以通过ElasticSearch自带的基于JSON REST API来进行交互。我们可以使用官方教程中提供的curl工具，或者稍微复杂一点的常用工具Fiddler或者RESTClient来进行交互，不过这里推荐使用[Sense](https://chrome.google.com/webstore/detail/sense-beta/lhjgkmllcaadmopgmanpapmpjgmfcfig)，这是Chrome内置的一个插件，能够提供很多的ElasticSearch的自动补全功能。
![](http://joelabrahamsson.com/PageFiles/669/1_sense_default_query.png)
![](http://joelabrahamsson.com/PageFiles/669/2_query_error.png)
当我们直接访问根目录时，会得到如下的提示:
```
{
   "name": "Mister Fear",
   "cluster_name": "elasticsearch",
   "version": {
      "number": "2.3.5",
      "build_hash": "90f439ff60a3c0f497f91663701e64ccd01edbb4",
      "build_timestamp": "2016-07-27T10:36:52Z",
      "build_snapshot": false,
      "lucene_version": "5.5.0"
   },
   "tagline": "You Know, for Search"
}
```

## CRUD
### Index:创建与更新索引
在ElasticSearch中，Index这一动作类比于CRUD中的Create与Update，当我们尝试为某个不存在的文档建立索引时，会自动根据其类似与ID创建新的文档，否则就会对原有的文档进行修改。ElasticSearch使用PUT请求来进行Index操作，你需要提供索引名称、类型名称以及可选的ID，格式规范为:`http://localhost:9200/<index>/<type>/[<id>]`。其中索引名称可以是任意字符，如果ElasticSearch中并不存在该索引则会自动创建。类型名的原则很类似于索引，不过其与索引相比会指明更多的细节信息：
- 每个类型有自己独立的ID空间
- 不同的类型有不同的映射(Mappings)，即不同的属性/域的建立索引的方案
- 尽可能地在一起搜索请求中只对某个类型或者特定的类型进行搜索

典型的某个Index请求为:
```
curl -XPUT "http://localhost:9200/movies/movie/1" -d'
{
    "title": "The Godfather",
    "director": "Francis Ford Coppola",
    "year": 1972
}'
```
在上述请求执行之后，ElasticSearch会为我们创建索引名为Movies，类型名为Movie，ID为1的文档。当然你也可以在Sense中运行该请求，这样的话用户体验会更好一点：
![](http://joelabrahamsson.com/PageFiles/669/4_sense_first_indexing_result.png)
在上图中我们可以了解到，ElasticSearch对于PUT请求的响应中包含了是否操作成功、文档编号等信息。此时我们如果进行默认的全局搜索，可以得到如下返回：
![](http://joelabrahamsson.com/PageFiles/669/6_sense_search_result.png)
可以看出我们刚刚新建的文档已经可以被查询，接下来我们尝试对刚才新建立的文档进行些修改，添加某些关键字属性。我们同样可以利用PUT请求来进行该操作，不过我们这次务必要加上需要修改的文档的ID编号:
```
curl -XPUT "http://localhost:9200/movies/movie/1" -d'
{
    "title": "The Godfather",
    "director": "Francis Ford Coppola",
    "year": 1972,
    "genres": ["Crime", "Drama"]
}'
```
对于此操作的ElasticSearch的响应与前者很类似，不过会可以看出`_version`属性值已经发生了变化：
![](http://joelabrahamsson.com/PageFiles/669/7_sense_update_result.png)
该属性即是用来追踪文档被修改过的次数，可以在乐观并发控制策略中控制并发修改，ElasticSearch仅会允许版本号高于原文档版本号的修改发生。

### GET
最简单的获取某个文档的方式即是基于文档ID进行搜索，标准的请求格式为:`http://localhost:9200/<index>/<type>/<id>`，我们查询下上文中插入的一些电影数据:
```
curl -XGET "http://localhost:9200/movies/movie/1" -d''
```
![](http://joelabrahamsson.com/PageFiles/669/8_sense_get_result.png)
返回数据中同样会包含版本信息、ID编号以及源信息。

### Delete:删除索引
现在我们尝试去删除上文中插入的部分文档，对于要删除的文档同样需要传入索引名、类型名与文档名这些信息，譬如:
```
curl -XDELETE "http://localhost:9200/movies/movie/1" -d''
```
![](http://joelabrahamsson.com/PageFiles/669/9_sense_delete_result.png)
在我们删除了该文档之后，再次尝试用GET方法获取该文档信息时，会得到如下的响应:
![](http://joelabrahamsson.com/PageFiles/669/10_sense_404.png)

## Search
ElasticSearch最诱人的地方即是为我们提供了方便快捷的搜索功能，我们首先尝试使用如下的命令创建测试文档:
```
curl -XPUT "http://localhost:9200/movies/movie/1" -d'
{
    "title": "The Godfather",
    "director": "Francis Ford Coppola",
    "year": 1972,
    "genres": ["Crime", "Drama"]
}'

curl -XPUT "http://localhost:9200/movies/movie/2" -d'
{
    "title": "Lawrence of Arabia",
    "director": "David Lean",
    "year": 1962,
    "genres": ["Adventure", "Biography", "Drama"]
}'

curl -XPUT "http://localhost:9200/movies/movie/3" -d'
{
    "title": "To Kill a Mockingbird",
    "director": "Robert Mulligan",
    "year": 1962,
    "genres": ["Crime", "Drama", "Mystery"]
}'

curl -XPUT "http://localhost:9200/movies/movie/4" -d'
{
    "title": "Apocalypse Now",
    "director": "Francis Ford Coppola",
    "year": 1979,
    "genres": ["Drama", "War"]
}'

curl -XPUT "http://localhost:9200/movies/movie/5" -d'
{
    "title": "Kill Bill: Vol. 1",
    "director": "Quentin Tarantino",
    "year": 2003,
    "genres": ["Action", "Crime", "Thriller"]
}'

curl -XPUT "http://localhost:9200/movies/movie/6" -d'
{
    "title": "The Assassination of Jesse James by the Coward Robert Ford",
    "director": "Andrew Dominik",
    "year": 2007,
    "genres": ["Biography", "Crime", "Drama"]
}'
```
这里需要了解的是，ElasticSearch为我们提供了通用的`_bulk`端点来在单请求中完成多文档创建操作，不过这里为了简单起见还是分为了多个请求进行执行。ElasticSearch中搜索主要是基于`_search`这个端点进行的，其标准请求格式为:`<index>/<type>/_search`，其中index与type都是可选的。换言之，我们可以以如下几种方式发起请求:
- **http://localhost:9200/_search** - 搜索所有的Index与Type
- **http://localhost:9200/movies/_search** - 搜索Movies索引下的所有类型
- **http://localhost:9200/movies/movie/_search** -仅搜索包含在Movies索引Movie类型下的文档

### 全文搜索
ElasticSearch的Query DSL为我们提供了许多不同类型的强大的查询的语法，其核心的查询字符串包含很多查询的选项，并且由ElasticSearch编译转化为多个简单的查询请求。最简单的查询请求即是全文检索，譬如我们这里需要搜索关键字:`kill`:
```
curl -XPOST "http://localhost:9200/_search" -d'
{
    "query": {
        "query_string": {
            "query": "kill"
        }
    }
}'
```
![](http://joelabrahamsson.com/PageFiles/669/11_sense_query_string_query.png)
执行该请求可能得到如下响应:
![](http://joelabrahamsson.com/PageFiles/669/12_search_result.png)


### 指定域搜索
在上文简单的全文检索中，我们会搜索每个文档中的所有域。而很多时候我们仅需要对指定的部分域中文档进行搜索操作，譬如我们要搜索仅在标题中出现`ford`字段的文档:
```
curl -XPOST "http://localhost:9200/_search" -d'
{
    "query": {
        "query_string": {
            "query": "ford",
            "fields": ["title"]
        }
    }
}'
```
![](http://joelabrahamsson.com/PageFiles/669/13_sense_with_fields.png)
而在全文搜索中，fields字段即被设置为了默认的`_all`值：
![](http://joelabrahamsson.com/PageFiles/669/14_sense_without_fields.png)

# Web Interface
## [Kibana](https://github.com/elastic/kibana)
> - [Kibana 4 权威指南](http://www.code123.cc/docs/kibana-logstash/v4/index.html)

![](http://7xkt0f.com1.z0.glb.clouddn.com/44798568-3E79-46F1-9546-D8BC0611E79F.png)

## [ElasticHQ](http://www.elastichq.org/index.html)
![](http://www.elastichq.org/img/screenie.png)

## [elasticsearch-head](https://github.com/mobz/elasticsearch-head)
![](http://mobz.github.io/elasticsearch-head/screenshots/clusterOverview.png)

![](http://153.3.251.190:11900/ElasticSearch)

