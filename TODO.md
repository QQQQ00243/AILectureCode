## 任务1: 使用PyTorch创建DNN，训练、评估DNN模型

### 1. 使用简单全连接网络作为分类器（Sigmoid激活函数），完成MNIST分类任务

* ~~训练~~
* ~~绘制错误例子~~
* ~~每个类的准确率~~
* ~~混淆矩阵~~

### 2.   使用深度全连接网络（2个隐藏层以上、Sigmoid激活函数）作为分类器，完成MNIST分类任务

### 3.   使用深度全连接网络（2个隐藏层以上）作为分类器，尝试不同激活函数，添加dropout，正则化等技巧（以上技巧不一定可以同时使用），完成MNIST分类任务

* ~~首先实现EarlyStop~~
* ~~探究BatchNorm，EarlyStop，dropout对模型性能的影响~~

### 4.   重复上述任务，完成MNIST-fashion分类任务

* ~~绘制数据集例子~~
* ~~简单全连接网络-Sigmoid~~
* ~~深度全连接网络（2个隐藏层以上、Sigmoid激活函数）~~
* ~~深度全连接网络（2个隐藏层以上）作为分类器，尝试不同激活函数，添加dropout，正则化等技巧~~

## 任务2: 使用PyTorch创建CNN，训练、评估CNN模型，使用CNN完成图像分类

### 1.使用LetNet5，完成MNIST分类任务，比对任务1中模型性能

### 2. 使用LetNet5，完成MNIST-fashion分类任务，比对任务1中模型性能

### 3.使用三通道LetNet5，完成Cifar10分类任务（有GPU可完成Cifar100分类）

* ~~三通道LetNet训练及评估~~
* ~~学习率时间表~~
* 实现图像增强

pytorch中num_workers详解https://blog.csdn.net/qq_24407657/article/details/103992170

### 4.使用ResNet作为分类器，完成Cifar10分类任务（有GPU可完成Cifar100分类）

### 5.使用VGG或ResNet预训练模型，微调后作为分类器，完成Cifar10分类任务（有GPU可完成Cifar100分类）

## 任务3: 使用PyTorch构建、改进CNN模型，完成特定计算机视觉任务
电阻、电容缺陷识别（见赛题说明）
要求设计或改进现有CNN模型，综合运用数据增强、正则化等技巧提高模型性能，使用多种评估指标测试模型性能，使用多种方法展示模型结构、复杂度、模型性能。

## 问题

* LeNet和全连接相比较在FashionMNIST数据集上表现区别不大
* Normalize 选取数值
* 调整optimizer参数
