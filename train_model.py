# -*- coding: utf-8 -*-

from NeuralNetwork import NeuralNetwork
import numpy
import pickle
import scipy.ndimage.interpolation

# 模型参数
input_nodes = 784   # 输入的图像是28*28像素的特征
hidden_nodes = 200  # 越多的隐藏层节点, 意味着更多的权重需要学习,模型精度更好,运算更慢
output_nodes = 10   # 输出有0-9种数字，信号最大的就是预测的分类
learning_rate = 0.1 # 更多的重复训练则使用更小的学习率（也就是下降的更慢）

# 重复训练的次数
epochs = 2

# 模型
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 加载训练数据
training_data_file = open('./mnist_dataset/mnist_train.csv', 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# 多轮重复训练, 有助于梯度下降的过程
for e in range(epochs):
    # 训练模型
    for record in training_data_list:
        all_values = record.split(',')

        # 为了避免S函数过饱和, 所以将像素数值归一到[0,1]
        # 为了避免0输入导致权重无法更新，所以最小不能为0
        inputs = (numpy.asfarray(all_values[1:]) / 255 * 0.99) + 0.01

        # 因S激活函数永远不会等于0与1, 所以需要对输出做预处理
        targets = numpy.zeros(10) + 0.01
        targets[int(all_values[0])] = 0.99

        # 输入给模型训练
        n.train(inputs, targets)

        # 左旋10度, 再喂进去训练一次, cval是旋转后的空白区域填充默认值
        #inputs_plus10 = scipy.ndimage.interpolation.rotate(inputs.reshape(28, 28), 20, cval = 0.01, reshape=False).reshape(28 * 28)
        #n.train(inputs_plus10, targets)

        # 右旋10度, 再喂进去训练一次, cval是旋转后的空白区域填充默认值
        #inputs_minus10 = scipy.ndimage.interpolation.rotate(inputs.reshape(28, 28), -20, cval = 0.01,  reshape=False).reshape(28 * 28)
        #n.train(inputs_minus10, targets)

# 加载测试数据
test_data_file = open('./mnist_dataset/mnist_test.csv', 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# 测试模型
scorecard = []
for record in test_data_list:
    all_values = record.split(',')
    # 正确的数字
    correct_label = int(all_values[0])
    # 输入预处理
    inputs = (numpy.asfarray(all_values[1:]) / 255 * 0.99) + 0.01
    # 模型预测
    outputs = n.query(inputs)
    # 得到最大输出节点的下标
    label = numpy.argmax(outputs)
    # 统计得分
    if label == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)

# 统计精度
scorecard_array = numpy.asarray(scorecard)
print("精度(性能):", scorecard_array.sum() / scorecard_array.size)

# 保存模型
with open('./n.model', 'wb') as fp:
    pickle.dump(n, fp)
