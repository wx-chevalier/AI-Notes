# -*- coding:utf-8 -*-
import logging
import logging.config
import ConfigParser
import numpy as np
import random
import codecs
import os
import LDAModel
import Preprocessor


def run():

    # 创建预处理器对象
    dpre = Preprocessor.Preprocessor()

    # 执行数据预处理
    dpre.preprocessing()

    # 创建LDA处理模型
    lda = LDAModel.LDAModel(dpre)

    # 执行参数估算
    lda.est()


if __name__ == '__main__':
    run()
