# -*- coding: utf-8 -*-

# 加载模型, 提供服务

import pickle
import  numpy
from io import BytesIO
import scipy.misc

class ModelServer:
    def __init__(self, model_file):
        with open(model_file, 'rb') as fp:
            self.model = pickle.load(fp)

    # 数字分类方法
    def predict(self, img):
        img_array = scipy.misc.imread(BytesIO(img), flatten=True)
        img_array = scipy.misc.imresize(img_array, (28, 28))

        img_data = 255.0 - img_array.reshape(28 * 28)
        img_data = (img_data / 255.0 * 0.99) + 0.01

        # 输入给模型, 得到输出
        outputs = self.model.query(img_data)
        # 得到最大输出节点的下标
        label = numpy.argmax(outputs)
        return label
