# Pj1
# 函数文件说明

neuralNetwork.m:训练神经网络

model_selection.m:遍历超参数选择对验证集最好的模型，保存参数及模型

best_model.m：对最好在测试集上预测，得到误差

MLPclassificationLoss.m：返回损失函数的梯度

MLPclassificationPredict.m：对验证、测试数据预测

standardizeCols.m：对数据进行标准化

# 模型训练（对随机初始化参数进行训练）
neuralNetwork.m

# 参数查找
model_selection.m

对学习率、正则化强度、模型大小进行查找
# 模型测试
best_model.m
