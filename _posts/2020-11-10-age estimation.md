---
layout: post
title: 论文笔记_age estimation----CORAL(2020)
date: 2020-11-10
tag: face

---

## age estimation----CORAL(2020)

### 经典算法

1. OR-CNN（Ordinal Regression CNN， 2016)

   将 K 阶 的序列回归问题 转换为 K-1 个 二分类任务，第 i 个任务的目标是预测年龄是否超过 $r_i$ 。所有的K-1个任务共享中间层，仅在最后一层拥有不同的权重。

   问题：不能保证一致性预测，每个子任务的输出可能出现冲突。

2. Ranking-CNN

   融合多个CNN二分类网络，综合所有预测值，得到年龄估计值。

### CORAL

提出CORAL算法，基于OR_CNN，但保证了输出的一致性（不会出现 大于20，小于15 这种输出），从而提升了估计精度。
保持一致性历史上有：cost_based weight on each task的方法，但该方法操作复杂且计算量大。CORAL算法，操作非常简单：==**保持最后的分类层，每个任务的weight都相同，只有bias不同**==。
论文证明了这样的约束可以保证输出一致性。

### 实验

- 在三个数据集MORPH-2，AFAD，CACD上，CORAL-CNN都优于CE-CNN（交叉熵分类loss）, OR-CNN。

- 统计了OR-CNN在预测上的不一致性：在错判上的不一致程度要严重于正判，这也说明了一致性对于预测精度确实有作用。



- Ranking-CNN 【