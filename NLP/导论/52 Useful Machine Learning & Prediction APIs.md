
> [52 个有用的机器学习与预测 API 盘点]()翻译自 [52 Useful Machine Learning & Prediction APIs](http://www.kdnuggets.com/2017/02/machine-learning-data-science-apis-updated.html)

 
# 52 个有用的机器学习与预测 API 盘点

## 摘要
本文翻译自 Thuy T. Pham 发布的 [52 Useful Machine Learning & Prediction APIs](http://www.kdnuggets.com/2017/02/machine-learning-data-science-apis-updated.html)，经原作者授权由InfoQ中文站翻译并分享。本文介绍了超过 50 个机器学习、预测、文本分析与分类、人脸识别、文本翻译等等领域的开放 API  ，这些 API 来源于 IBM、Google 以及 Microsoft 这些在人工智能领域迅猛发展的公司，希望能够帮助开发者让他们的应用更加的智能。


## 正文
随着基于人工智能与机器学习的应用如雨后春笋般不断涌现，我们也看到有很多提供类似功能的 API 悄悄登上了舞台。 API 是用于构建软件应用的程序、协议以及工具的组合；本文是对[2015 中这个列表](http://www.kdnuggets.com/2015/12/machine-learning-data-science-apis.html/2)的修正与完善，移除了部分被废弃的 API ；我们也添加了最近由 IBM、Google、Microsoft 这些大厂发布的 API 。所有的 API 可以根据应用场景进行分组：
- 人脸与图片识别。
- 文本分析，自然语言处理以及情感分析。
- 语言翻译。
- 预测以及其他的机器学习算法。


在具体的每个分组内，我们根据首字母顺序排序； API 的描述信息源自截止到 2017 年 2 月 3 日对应主页上的描述。如果你发现存在未被收录的流行的 API 可以在评论中留言，我们会将其增补到列表中。


# 人脸与图片识别
1. [Animetrics Face Recognition:](http://api.animetrics.com/) 该 API 能够用于图片中的人脸检测，并且将其与已知的部分人脸进行匹配分析；该 API 还支持从某个待搜索的集合中添加或者移除某个分类，或者从某个分类中添加或者删除某张人脸图片。
2. [Betaface](https://www.betaface.com/wpa/):  同样是提供人脸识别与检测的在线服务。它支持多人脸检测、人脸裁剪、123 个人脸特征点提取、人脸验证、识别以及大型数据库中的相似性搜索提取。
3. [Eyedea Recognition: ](http://www.eyedea.cz/)致力于提供高阶的计算机视觉解决方案，主要包括对象检测与识别。其识别服务提供了常见的眼部、人脸、车辆、版权以及果盘识别，该 API 主要的价值在于对于对象、用户以及行为的快速识别。
4. [Face++](https://www.faceplusplus.com/): 为应用提供面部的检测、识别以及分析服务，用户可以通过 API 调用训练模型，进行人脸检测、人脸识别、人脸分类、图像修正、创建人脸分组等等服务。
5. [FaceMark](http://apicloud.me/apis/facemark/docs/):  提供了能够在正面照片中检测 68 个特征点以及侧面照片中检测 35 个特征点的服务。
6. [FaceRect](http://apicloud.me/apis/facerect/demo/): 提供了非常强力与完整的面部检测的 API ，包括在正面照片与侧面照片中检测面部以及在单张照片中提取多个面部的功能；它还能将结果以 JSON 格式输出，包括检测到的眼睛、鼻子、嘴等等面部特征。
7. [Google Cloud Vision API](https://cloud.google.com/vision/): 架构于著名的 [TensorFlow](http://www.tensorflow.org/) 之上，能够高效地学习与预测图片中的内容。它能够有助于用户搜索最爱的图片，并且获取图片中丰富的注释。它还能将图片按照船、狮子、埃菲尔铁塔等等不同的类别进行分类，并且对照片中不同表情的面部进行识别，除此之外它还能将图片中不同国家的语言打印出来。
8. [IBM Watson Visual Recognition](https://www.ibm.com/watson/developercloud/visual-recognition.html): 该  API 能够辅助理解图片内容，包括图片标记、人脸识别、年龄估计以及性别判断，还能根据人脸相似度进行搜索。开发者能够在该服务的基础上结合自身业务特点定制出各式各样奇妙的产品。
9. [Kairos](https://www.kairos.com/docs/api/): 该平台方便开发者快速添加 [情感分析](https://www.kairos.com/emotion-analysis-api) 与 [人脸识别](https://www.kairos.com/face-recognition-api) 的功能到应用与服务中。
10. [Microsoft Cognitive Service - Computer Vision](https://www.microsoft.com/cognitive-services/en-us/computer-vision-api): 该 API 能够根据用户输入与用户选择分析可视化内容。譬如根据内容来标记图片、进行图片分类、人类识别并且返回他们的相似性、进行领域相关的内容识别、创建图片的内容描述、定位图片中的文本、对图片内容进行成人分级等。
11. [Rekognition](http://www.programmableweb.com/api/rekognition): 该 API 能够根据社交图片应用的特点提供快速面部识别与场景识别。譬如基于人眼、嘴、面部以及鼻子等等特征进行性别、年龄以及情绪预测。

12. [Skybiometry Face Detection and Recognition](https://skybiometry.com/Documentation/): 该 API 提供人脸检测与识别服务，新版本的 API 还提供了深色微分功能。


# 文本分析，自然语言处理，情感分析


1. [Bitext ](https://www.bitext.com/text-analysis-api-2/#How-accurate-is-the-analysis)提供了目前市场上来说几乎最准确的基于情感的多主题识别，目前包括四个语义服务：实体与概念抽取、情感分析与文本分类；该 API 总共支持 8 种不同的语言。
2. [Diffbot Analyze](https://www.diffbot.com/dev/docs/analyze/): 为开发者提供了从任何网页中识别、分析以及提取主要内容与区块的功能。
3. [Free Natural Language Processing Service](https://market.mashape.com/loudelement/free-natural-language-processing-service): 提供了包括情感分析、内容提取以及语言检测等功能。它同样是 mashape.com 这个大型云 API 市场中的畅销产品之一。
4. ![new](http://www.kdnuggets.com/images/newr.gif) [Google Cloud Natural Language API](https://cloud.google.com/natural-language/reference/rest/): 该 API 提供了对于文档的架构与含义进行分析的功能，包括情感分析、实体识别以及文本标注等。
5. [IBM Watson Alchemy Language](http://www.alchemyapi.com/): 该 API 能够辅助电脑学习如何阅读以及进行一些文本分析任务。譬如将非结构化数据按照领域模型转化为结构化数据，使其能够服务于社交媒体监测、商业智能、内容推荐、商业交易以及定向广告等等服务。
6. [MeaningCloud Text Classification](https://www.meaningcloud.com/developer/text-classification): 该 API 提供了部分预分类的功能：文本提取、分词、停用词设置以及同义词提取等等。
7. ![new](http://www.kdnuggets.com/images/newr.gif) [Microsoft Azure Text Analytics API](https://docs.microsoft.com/en-us/azure/machine-learning/machine-learn) 基于 Azure Machine Learning 实现的一系列文本分析服务。该 API 能够用于情感分析、关键语句提取、语言检测以及主题识别这些非结构化文本的处理任务。该 API 并不需要使用者提供相关的训练数据，能够大大降低使用门槛。
8. [Microsoft Cognitive Service - Text Analytics](https://www.microsoft.com/cognitive-services/en-us/text-analytics-api): 提供了情感检测、关键语句提取、主题以及语言分析等功能。该分组中其他的 API 还包括 [Bing 拼写检测](https://www.microsoft.com/cognitive-services/en-us/bing-spell-check-api)、[语言理解](https://www.microsoft.com/cognitive-services/en-us/language-understanding-intelligent-service-luis)、[文本分析](https://www.microsoft.com/cognitive-services/en-us/linguistic-analysis-api)、[Web 语言模型](https://www.microsoft.com/cognitive-services/en-us/web-language-model-api)等等。
9. [nlpTools](http://nlptools.atrilla.net/web/api.php): 简单的采用 JSON 传输格式的提供了自然语言处理功能的 HTTP RESTful 服务。它能够提供对于在线媒体的情感分析与文本分类等服务。
10. [Semantic Biomedical Tagger](http://docs.s4.ontotext.com/display/S4docs/Semantic+Biomedical+Tagger): 能够利用文本分析技术提取出文档中的 133 个生物医药学相关的实体词汇并且将它们链接到知识库中。
11. [Thomson Reuters Open Calais™](http://www.opencalais.com/opencalais-api/): Calais 基于自然语言处理与机器学习技术，能够分类与关联文档中的实体信息（人名、地名、组织名等）、事实信息（员工 x 为公司 y 工作）、事件信息（员工 z 在 x 日被任命为 y 公司的主席） 。
12. [Yactraq Speech2Topics](http://yactraq.com/) 提供了基于语音识别与自然语言处理技术的将语音内容转化为主题数据的云服务。



# 语言翻译


1. [Google Cloud Translation](https://cloud.google.com/translate/docs/): 能够在数以千计的语言之间完成文本翻译工作。该 API 允许网页或者程序方便地接入这些翻译服务。
2. ![new](http://www.kdnuggets.com/images/newr.gif) [IBM Watson Language Translator](http://www.ibm.com/watson/developercloud/language-translator.html):  能够在不同语言之间进行文本翻译，该服务允许开发者基于独特的领域术语与语言特性进行自定义模型开发。
3. [LangId](http://langid.net/identify-language-from-api.php): 能够快速地从多语言中检索结果的服务，并不需要使用者指定哪种语言，并且能够返回结果对应的语言类型。
4. ![new](http://www.kdnuggets.com/images/newr.gif) [Microsoft Cognitive Service - Translator](https://www.microsoft.com/cognitive-services/en-us/text-analytics-api): 能够自动地在翻译之前进行语言类型检测，支持 9 种语言的语音翻译以及 60 种语言的文本翻译。
5.  [MotaWord](https://www.motaword.com/developer): 快速地人工翻译平台，提供了超过 70 种语言支持。该 API 同样允许开发者查询翻译报价、上传带有文档说明与样式指南的翻译项目请求、自动追踪翻译进度以及进行实时反馈等。
6.  [WritePath Translation](https://www.writepath.co/en/developers):  API 允许开发者将 WritePath 功能集成到自定义应用中，包括字数检索、提交文本翻译任务、以及获取翻译信息等等。


# 预测与其他机器学习 API 


1. [Amazon Machine Learning](https://aws.amazon.com/documentation/machine-learning/): 寻找数据中的隐藏模式信息，典型的用法包括诈骗检测、天气预报、市场营销以及点击预测等。
2. [BigML](https://bigml.com/api/): 提供基于云的机器学习与数据分析服务，允许用户以 HTTP 请求的方式自己创建数据源以及选择合适的模型来处理有监督或者无监督的机器学习任务。
3. [Ersatz](http://www.ersatzlabs.com/documentation/api/): 基于 GPU 支持的深度神经网络提供的预测服务，允许用户以 API 方式进行交互。Ersatz 中还利用增强学习来合并不同的神经网络模型来提升整体的效果。
4. [Google Cloud Prediction](https://cloud.google.com/prediction/docs/): 提供了用于构建机器学习模型的 RESTful  API 。这些工具能够通过分析数据来提取出应用中数据的不同特征，譬如用户情感、垃圾信息检测、推荐系统等等。
5. ![new](http://www.kdnuggets.com/images/newr.gif) [Google Cloud Speech API](https://cloud.google.com/speech/docs/apis): 能够提供超过 80 种语言的快速与准确的语音识别以及转化服务。
6. [Guesswork.co](http://www.guesswork.co/): 能够为电商网站提供产品推荐引擎，Guesswork 可以通过基于 Google 预测 API 构建的语义化引擎来对用户行为进行预测。
7. [Hu:toma:](https://www.hutoma.com/about.html) 帮助世界各地的开发者构建商用级别的深度学习聊天机器人。
8. ![new](http://www.kdnuggets.com/images/newr.gif) [IBM Watson Conversation  ](https://www.ibm.com/watson/developercloud/conversation.html): 帮助构建可以部署在多个消息平台或者网页上的，能够理解自然语言的聊天机器人。其他类似的 API 还包括 [Dialog](https://www.ibm.com/watson/developercloud/dialog.html)、[Natural Language Classifier](https://www.ibm.com/watson/developercloud/nl-classifier.html)、[Personality Insights](https://www.ibm.com/watson/developercloud/personality-insights.html)、[Document Conversion](https://www.ibm.com/watson/developercloud/document-conversion.html) 以及 [Tone Analyzer](https://www.ibm.com/watson/developercloud/tone-analyzer.html).
9. ![new](http://www.kdnuggets.com/images/newr.gif)  [IBM Watson Speech  ](https://www.ibm.com/watson/developercloud/speech-to-text.html): 包含了 [语音到文本](https://www.ibm.com/watson/developercloud/speech-to-text.html) 以及 [文本到语音](https://www.ibm.com/watson/developercloud/text-to-speech.html) 之间的转化功能（譬如创建语音控制的应用）。
10. ![new](http://www.kdnuggets.com/images/newr.gif) [IBM Watson Data Insights](https://www.ibm.com/watson/): 该系列的服务包含了三个 API ：AlchemyData News、Discovery 以及 Tradeoff Analytics。AlchemyData 提供了对于大量的新闻、博客内容的高级别定向搜索与趋势分析的服务。Tradeoff Analytics 则是帮助用户在多目标优化时进行有效抉择。
11. [IBM Watson Retrieve and Rank](http://www.ibm.com/watson/developercloud/retrieve-rank.html):  开发者可以将自定义数据导入到服务中，并且使用相关的关联发算法来训练机器学习模型。服务的输出包括了一系列相关的文档与元数据，譬如某个联络中心的代理能够基于该服务提高呼叫的平均处理时间。
12. [Imagga](https://imagga.com/solutions/auto-tagging.html): 能够为你的图片自动打标签，从而允许你的图片可以被关联搜索到。
13. [indico](https://indico.io/docs): 提供了文本分析（情感分析、Twitter 参与度、表情分析等）以及 图片分析（面部表情识别、面部定位）。indico  的 API 可以免费试用并且不需要任何的训练数据。
14. [Microsoft Azure Cognitive Service ](https://azure.microsoft.com/en-au/services/cognitive-services/)  API  : 基于预测分析提供机器学习推荐服务，譬如个性化产品推荐等，可以用来代替传统的 Azure Machine Learning Recommendations 服务。新版本提供了批处理支持，更好地 API 检索服务、更清晰的 API 使用界面以及更好的注册与账单界面等。
15. ![new](http://www.kdnuggets.com/images/newr.gif)  [Microsoft Azure Anomaly Detection API](https://gallery.cortanaintelligence.com/MachineLearningAPI/Anomaly-Detection-2) : 能够在序列数据中检测出异常数据，譬如检测内存使用过程中是否存在内存泄露的情况。
16. ![new](http://www.kdnuggets.com/images/newr.gif) [Microsoft Cognitive Service - QnA Maker](https://www.microsoft.com/cognitive-services/en-us/qnamaker): 将信息提取为会话式的、易于浏览的数据形式。其他类似的 API 还包括 [Academic Knowledge](https://www.microsoft.com/cognitive-services/en-us/academic-knowledge-api)、[Entity Linking](https://www.microsoft.com/cognitive-services/en-us/entity-linking-intelligence-service)、[Knowledge Exploration](https://www.microsoft.com/cognitive-services/en-us/knowledge-exploration-service)以及[Recommendations](https://www.microsoft.com/cognitive-services/en-us/recommendations-api)。
17. ![new](http://www.kdnuggets.com/images/newr.gif) [Microsoft Cognitive Service - Speaker Recognition](https://www.microsoft.com/cognitive-services/en-us/speaker-recognition-api): 帮助应用来分析检测出当前的发言者。其他的类似于的 API 还包括[Bing Speech](https://www.microsoft.com/cognitive-services/en-us/speech-api) （将语音转化为文本并且理解其大致含义）、 [Custom Recognition](https://www.microsoft.com/cognitive-services/en-us/custom-recognition-intelligent-service-cris) 等等。
18. [NuPIC](https://github.com/numenta/nupic/wiki/NuPIC-API---A-bird's-eye-view) : 由 NuPIC 社区运行与维护的开源项目，其基于 Python/C++ 实现了 Numenta's Cortical Learning 算法并对外提供 API 服务。该 API 允许开发者能够使用基本算法或者分层算法，也可以选择使用其他的平台功能。
19. [PredicSis](https://predicsis.ai/): 能够通过预测分析与大数据技术提供市场营销的效用与收益。
20. [PredictionIO](http://predictionio.incubator.apache.org/index.html): 基于 Apache Spark、HBase 以及 Spray 这些著名的开源项目搭建的开源机器学习服务。典型的 API 包括了创建与管理用户信息及其行为记录、检索项目与内容、基于用户进行个性推荐等等。
21. [RxNLP - Cluster Sentences and Short Texts](http://www.rxnlp.com/api-reference/cluster-sentences-api-reference/): 提供了文本挖掘与自然语言处理的服务。其中语句聚类 API 能够将不同的语句进行分类，譬如将不同新闻文章中的语句或者 Twitter、Facebook 上提取出来的短文本划分到不同的分组中。
22. [Sightcorp F.A.C.E.](http://face.sightcorp.com/doc_swagger/): 该 API 能够帮助第三方应用来更好地理解用户行为，并且根据年龄、性别、面部表情、头部姿势以及种族划分来进行相似面部的分析与搜索。


> 本文已获得原作者翻译授权，查看原文：[52 Useful Machine Learning & Prediction APIs](http://www.kdnuggets.com/2017/02/machine-learning-data-science-apis-updated.html)


