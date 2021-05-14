# 螺旋桨RNA结构预测竞赛第一名方案

## 项目描述

### 竞赛要求
基于百度发布的RNA二级结构预测算法LinearFold和RNA配分方程算法LinearPartition，预测给定RNA序列在每个位点上保持不成对的概率.

详细请见： https://aistudio.baidu.com/aistudio/competition/detail/61

## 训练数据: 

### 输入
   1. 5000个RNA序列, 长度较均匀分布在80和500之间
   2. Linear_fold计算的二级结构 (dot-bracket notation)
   3. Linear_partition计算的碱基对概率
    
### 标签
   每个位点的不成对概率
    
## 模型:

网络的设计主要考虑了三个支配RNA碱基配对的因素
   1. 来自于全部序列的排列组合（配分）竞争，用Attention机制来模拟
   2. 来自于线性大分子的一维序列限制， 用LSTM结构来模拟
   3. 来自于局部紧邻碱基的合作（比如，一个孤立的碱基对极不稳定）， 用1D Convolution来模拟
 
所以框架由以上三个模块组成， 并在输入和输出层加了1-3个线性层。除非特意说明， 所有的隐藏层的维度为32.

训练中发现高维度和深度的网络构架并不能给出更好的结果！

Three main mechanisms directing RNA base pairing are taken into consideration for the design of the network architecture. 
   1) The combinatorial configurational space of attainable RNA base pairs, approximated by Attention Mechanism
   2) The quasi-1D nature of unbranched, continuous RNA polymers, approximated by LSTM
   3) The cooperativity of neighboring bases for stable base pairing, approximated by 1D Convolution

Hence the neural net comprises of three main building blocks, in addition to linear layers for the input and output. 

The dimensions of all hidden layers are 32 unless noted otherwise.

Wider and/or deeper nets gave similar, but no better, performances!

### 网络框架:

 1. 输入矩阵 NLC, N=1: batch size, L=512: 序列长度, C=10: 通道数.
 2. 线性层 x 1
 3. TransformerEncoder x 1 (nhead=2)
 4. Bidirectional LSTM x 1
 5. 1D convolution x 1
 6. 线性输出层 x 3, 维度为32, 32, 2 

### 损失函数:
 softmax+bce 或 softmax+mse 给出相近的结果, 最后采用方法是训练用softmax+bce, 然后用softmax+mse验证.

### 优化方法:
 1. 优化器为adam, learning_rate=0.003 加 lr_scheduler (连续七次损失不降低, learning_rate减小10%)
 2. dropout=0.2 (鉴于网络较小)
 3. 无L1/L2正则化 (鉴于train/validation差异较小)
 4. 采用earlystop, 一个epoch检测10次, 如果validation loss降低, 存储模型


## 项目结构
```
-|checkpoint
-|work
   -|code
   -|data
-README.MD
-requirements.txt
-fly.ipynb
```
## 使用方式

详细使用方法请参照fly.ipynb中的文档内容.

A：在AI Studio上[运行本项目](https://aistudio.baidu.com/aistudio/projectdetail/1479469)

     需要拷贝work目录和fly.ipynb 保持原有目录结构，即可运行fly.ipynb

B：在本机运行方法如下：

    1) 克隆 github repo 到本地目录

    2) 安装所需函数库 (参照requirements.txt)
    
    3) 运行 fly.ipynb
