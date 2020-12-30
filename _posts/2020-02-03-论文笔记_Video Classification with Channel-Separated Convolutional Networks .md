---
layout: post
title: 论文笔记_Video Classification with Channel-Separated Convolutional Networks
---

[paper](https://arxiv.org/abs/1904.02811)  [code](https://github.com/facebookresearch/VMZ)

## 1. 摘要

组卷积(group convolution) 在2D卷积结构中能够很大程度地降低计算量。因此，本文研究3D卷积（视频分类问题）中组卷积的不同设计方式的影响。实验表明通道交互量对3D组卷积的精度有很大影响。实验发现：1. 将3D卷积在通道和时空域上分离会得到更高精度和更低计算量。2. 3D通道分离卷积属于一种正则化，虽然会导致更低的训练精度但可以取得更高的测试精度。 结合实验，提出 通道分离卷积网络(Channel Separated Convolution Networks--CSN)，计算简单且精度匹敌目前state of the art 算法。

---

## 2. 背景

CSN与Xception类似：卷积都使用了通道分离的思想。 Xception 在**channel**和**space**上分解2D卷积， CSN在**channel**和**space-time**上分解3D卷积。 （CSN使用的是bottleneck blocks， 而Xception用的是普通的blocks）。

CSN的resnet变体ir-CSN与ResNeXt类似，差异是：ir-CSN 使用depthwise convolutions，ResNeXt使用group convolutions。

---

## 3.主要概念

**group convolution:**  将卷积按通道划分为G组，只有同一个组内的通道才互相连接，极端情况是组数G=通道数，即每组只有一个通道（depthwise conv)。组卷积会大大减少参数量和模型计算量，但同时也会降低通道之间的交互。

![image-20200208202603166](../images/image-20200208202603166.png)

(Xception, MobileNet是首先采用depthwise conv的网络)

**channel interactions 定量：**将通过任一输出神经元连接的两个神经元称为一对交互神经元。将输入层中交互神经元的对数作为衡量channel interactions的量。

对于一层3D卷积：卷积核大小为：k\*k\*k，输入的时空域向量的像素数为：THW。则有：

![image-20200208205732037](../images/image-20200208205732037.png)

其中：![image-20200208210123526](../images/image-20200208210123526.png)

**channel separation （通道分离）：** 通道分离卷积网络（CSN）定义为：除conv1以外，所有卷积层，均是1\*1\*1的传统卷积 或 k\*k\*k的depthwise conv。CSN网络将通道信息交互和位置信息交互进行了分离，1\*1\*1的卷积层用来进行通道信息交互， k\*k\*k的depthwise conv用来进行位置信息的交互。

2D中的Xception和MobileNet都属于通道分离网络。3D中的P3D，R(2+1)D，S3D网络都属于分离网络，但它们都是将空间域和时间域进行分离。

根据通道分离的程度CSN可以分为：**interaction-preserved channel-separated network (ip-CSN)--图b)**和**interaction-reduced channel-separated network (ir-CSN)--图c)**。 ip-CSN 用一个depthwise conv和一个1\*1\*1的通道交换层替换传统的conv层，减少计算量但仍保留一定程度的通道交换。ir-CSN 直接用一个depthwise conv替换传统conv，本层中去掉了通道信息交换，只保留了位置信息交换。

![image-20200208211704119](../images/image-20200208211704119.png)

---

## 4. 网络设计

将group conv应用于resnet中：

**Resnet simple block：**如下图三种组合形式：a) 原始simple block；b) group conv simple block; c) depthwise conv simple block。（省略BN, ReLU, skip connections）

![image-20200208212417054](../images/image-20200208212417054.png)

**Resnet bottleneck block：**如下图四种组合形式：a) 原始bottleneck block；b) group conv bottleneck block; c) depthwise conv bottleneck block；d) depthwise + group conv bottleneck block。（省略BN, ReLU, skip connections）

<img src="../images/image-20200209123335493.png" alt="image-20200209123335493" style="zoom: 67%;" />![image-20200209235650237](../images/image-20200209235650237.png)

设计理由：b) 是 ResNeXt block； c) 是b对应的depthwise 变体；d) 类似于ShuffleNet block。

---

## 5. Ablation Experiment

实验具体设置见论文。

**数据增强：** 空间：rescale+crop，时间：sample+skip

**基础网络：ResNet3D**

<img src="../images/image-20200209235829588.png" alt="image-20200209235829588" style="zoom:80%;" />

对应通道分离网络：

<img src="../images/image-20200210000100237.png" alt="image-20200210000100237" style="zoom:67%;" />



训练误差曲线：

<img src="../images/image-20200210000225972.png" alt="image-20200210000225972" style="zoom:67%;" />

ip-CSN因为通道信息交互少，因此训练误差大，但在测试集上表现更好。说明*ip-CSN具有正则化的作用，减少了过拟合。*

**核心实验1结果：**

![image-20200210000502926](../images/image-20200210000502926.png)

对比不同结构对应的精度。结论：**Bottleneck-D(ir-CSN)是计算量和精度的最好折中。**



<img src="../images/image-20200210001123684.png" alt="image-20200210001123684" style="zoom:67%;" />



**数据集测试：** 

*附录进行了网络可视化。*





