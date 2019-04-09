# -*- coding: utf-8 -*-

from ModelServer import *

# 加载模型服务
model_server = ModelServer('./n.model')

# 对手写的字识别率很低
for i in range(0, 10):
    with open('./handwriting/{}.png'.format(i), 'rb') as fp:
        num = model_server.predict(fp.read())
        print('数字:{}, 预测为:{}'.format(i, num))
