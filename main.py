# -*- coding: utf-8 -*-

from ModelServer import *

# 加载模型服务
model_server = ModelServer('./n.model')

# 对手写的字识别率很低
with open('./handwriting/3.png', 'rb') as fp:
    num = model_server.predict(fp.read())
    print('预测为:', num)
