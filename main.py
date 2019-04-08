# -*- coding: utf-8 -*-

from NeuralNetwork import NeuralNetwork

# 模型参数
input_nodes = 3
hidden_nodes = 4
output_nodes = 3
learning_rate = 0.5

# 模型
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 训练
n.train([1, 2, 3], [4, 5, 6])

# 预测
r = n.query([1, 2, 23])
#print(r)
