# 基于卷积神经网络的垃圾分类系统

## 后端使用Tornado

通过运行server.py启动后端

## 深度学习框架为Tensorflow，使用GPU训练

网络模型在net_struct中，模型包括

+ LeNet5
+ AlexNet
+ VGGNet
+ InceptionV3
+ ResNet

只需要简单修改数据集就可以训练自己的模型，模型最终保存到model文件夹下，log保存到logs文件夹下，log可以使用tensorboard进行查看。

模型参数略有改动，其中InceptionV3和ResNet采用迁移学习

