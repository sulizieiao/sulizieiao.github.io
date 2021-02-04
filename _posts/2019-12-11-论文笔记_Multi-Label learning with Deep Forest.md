---
layout: post
title: 论文笔记_Multi-Label learning with Deep Forest
date: 2019-12-11 
tag: DL基础
---

### 背景： 
多标签应用广泛（文本分类，场景分类，视频分类...）
1. 直接将多标签 转化为 多个单标签分类器： 计算量翻多倍；没有考虑标签之间的相关性。
2. 基于神经网络：top层改成multi-label。 BP-MLL-->entropy loss+deep neural networks。 在文本分类上表现好，但需要大量训练数据。

---->> gcforest 相比于 DNN, 同样可以学习到特征表述能力，并且更好训练，拥有更少的超参数。本文研究基于gcforest的多标签模型：MLDF

gcforest 核心： layer-by-layer feature transformation in an ensemble way.
multi-label核心： 学习使用标签之间的相关性。
两者结合--->MLDF(mulit-label deep forest) 使用多标签树作为深度森林中的基本单元，利用层到层的特征表述学习来利用标签之间的相关性。（A层关于标签M的预测，会作为特征传递到A+1层，用于其他标签的预测）

不同应用场景有不同的measure指标，为了让MLDF对应特定指标最优，提出了：measure-aware feature reuse（重复利用上一次中好的特征表述） 和 measure-aware layer growth（控制模型复杂度并抑制过拟合，基于measure自动停止层数增加）.
