# -*- coding: utf-8 -*-
import numpy
import scipy.special

# 神经网络实现

# 输入层 + 隐藏层 + 输出层
class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes    # 输入层节点个数
        self.hnodes = hiddennodes   # 隐藏层节点个数
        self.onodes = outputnodes   # 输出层节点个数
        self.lr = learningrate

        # 初始化权重矩阵：生成若干x值, 它们的分布密度符合正态曲线, 均值0代表以x=0对称, 标准差小于根号nodes数量
        # 数学知识链接：https://www.shuxuele.com/data/standard-normal-distribution-table.html

        # 输入层->隐藏层
        self.wih = numpy.random.normal(0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        # 隐藏层->输出层
        self.who = numpy.random.normal(0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

    @staticmethod
    def activation_function(x):
        # 定义激活函数为sigmoid
        return scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        # 输入列表转换成二维矩阵（就是平面上的矩阵）, 然后转置成单列
        inputs = numpy.array(inputs_list, ndmin = 2).T

        # 输出列表转换成二维矩阵（就是平面上的矩阵）, 然后转置成单列
        targets = numpy.array(targets_list, ndmin = 2).T

        # 阶段1: 信号前馈

        # 计算隐藏层的输入
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 隐藏层执行激活函数
        hidden_outputs = self.activation_function(hidden_inputs)

        # 计算输出层的输入
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 输出层执行激活函数
        final_outputs = self.activation_function(final_inputs)

        # 阶段2：误差反向传播

        # 计算输出层的误差
        output_errors = targets - final_outputs
        # 按权重矩阵传播误差到隐藏层
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # 阶段3：梯度下降调整权重

        # 更新隐藏层->输出层的权重矩阵
        self.who += numpy.dot(self.lr * output_errors * final_outputs * (1 - final_outputs), hidden_outputs.T)
        # 更新输入层->隐藏层的权重矩阵
        self.wih += numpy.dot(self.lr * hidden_errors * hidden_outputs * (1 - hidden_outputs), inputs.T)

    def query(self, inputs_list):
        # 输入列表转换成二维矩阵（就是平面上的矩阵）, 然后转置成单列
        inputs = numpy.array(inputs_list, ndmin = 2).T

        # 计算隐藏层的输入
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 隐藏层执行激活函数
        hidden_outputs = self.activation_function(hidden_inputs)

        # 计算输出层的输入
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 输出层执行激活函数
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
