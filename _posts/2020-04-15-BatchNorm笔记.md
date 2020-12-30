---
layout: post
title: BatchNorm 笔记

---

## BN算法

1. [ 基于mxnet的BN层原理和代码实现]([http://zh.gluon.ai/chapter_convolutional-neural-networks/batch-norm.html#%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AE%9E%E7%8E%B0](http://zh.gluon.ai/chapter_convolutional-neural-networks/batch-norm.html#从零开始实现))

2. Batch Normalization的加速收敛作用体现在两个方面(**2017年观点：可以防止产生"internal covariate shift"现象**）：一是归一化了每层和每维度的scale，所以**可以整体使用一个较高的学习率**，而不必像以前那样迁就小scale的维度；二是归一化后使得更多的权重分界面落在了数据中，**降低了overfit的可能性**，因此一些防止overfit但会降低速度的方法，例如dropout和权重衰减就可以不使用或者降低其权重。**争议**的重点：归一化的位置，还有gamma与beta参数的引入，从理论上分析，论文中的这两个细节实际上并不符合ReLU的特性：ReLU后，数据分布重新回到第一象限，这时是最应当进行归一化的；gamma与beta对sigmoid函数确实能起到一定的作用（实际也不如固定gamma=2），但对于ReLU这种分段线性的激活函数，并不存在sigmoid的低scale呈线性的现象。

3. [BN的真实作用](https://zhuanlan.zhihu.com/p/52132614)是：(**2018年观点：使得landscape变得平滑**）这种平滑使梯度更具可预测性和稳定性，从而使训练过程更快。论文认为：1. BN与ICS不相关，增加了ICS，BN的作用依然很好。2. BN并不能减少ICS，相反，还会有一定程度上的增加。3. BN并不是唯一使得landscape变平滑的方法，还可以设计更优的方法。下图表示resnet中的skip connection 也是因为使得landspcape变的平滑，而改进了模型。![image-20200415144019732](../images/image-20200415144019732.png)

---

### 具体实现：

![image-20200415113907005](../images/image-20200415113907005.png)

这里的最后一步也称之为仿射(affine)，引入这一步的目的主要是设计一个通道，使得输出output至少能够回到输入input的状态，使得BN的引入至少不至于降低模型的表现，这是深度网络设计的一个套路。

BN层中的`running_mean`和`running_var`的更新是在`forward()`操作中进行的，而不是`optimizer.step()`中进行的，因此如果处于训练状态，就算你不进行手动`step()`，BN的统计特性也会变化的（pytorch)

