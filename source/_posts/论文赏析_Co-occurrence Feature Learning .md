---
title: >-
  论文赏析_Co-occurrence Feature Learning from Skeleton Data for Action Recognition and
  Detection with Hierarchical Aggregation
date: 2018-12-04 08:07:46
tags: [论文, CV, IJCAI2018, 人体骨架检测, 论文赏析]
---

code: https://github.com/huguyuehuhu/HCN-pytorch

<!--more-->

### 摘要

解决这一任务的最关键因素在于两方面：用于关节共现的帧内表征和用于骨架的时间演化的帧间表征。

这些共现特征是用一种分层式的方法学习到的，其中不同层次的环境信息（contextual information）是逐渐聚合的。首先独立地编码每个节点的点层面的信息。然后同时在空间域和时间域将它们组合成形义表征。

具体而言，我们引入了一种全局空间聚合方案，可以学习到优于局部聚合方法的关节共现特征。

此外，我们还将原始的骨架坐标与它们的时间差异整合成了一种双流式的范式。

### Introduction

