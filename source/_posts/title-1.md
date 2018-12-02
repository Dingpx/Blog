---
layout: posts
title: CS231n Assignment2
date: 2018-12-03 00:35:55
tags: [cs231n, assignment]
---



听完了整体的CNN的框架和常用的调参技巧后，开始着手做assignment2

以下记录做题时遇到的问题

<!--more-->

## FullyConnectedNets.ipynb

这个ipynb主要是将assignment1中的two_layer_net中实现的功能用一种模块化的形式去实现更复杂（多层网络）的结构

### 数据预处理

首先要运行一个setup.py文件，目的是将某些用C写的东西编译出来，主要是涉及一个imcool的算法，去减少卷积的算法复杂度（还没有细看，就先跑一下）

```python
# As usual, a bit of setup
from __future__ import print_function	##__future__兼容此版本没有的库
import time
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.fc_net import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver

# ipython里的magic function，用来内嵌绘图，同时可以省略plt.show()
%matplotlib inline	

# 默认的像素：[6.0,4.0]，分辨率为100，图片尺寸为 600&400
# 指定dpi=200，图片尺寸为 1200*800
# 指定dpi=300，图片尺寸为 1800*1200
# 设置figsize可以在不改变分辨率情况下改变比例
plt.rcParams['figure.figsize'] = (10.0, 8.0) 
plt.rcParams['image.interpolation'] = 'nearest' # 设置最近邻插值
plt.rcParams['image.cmap'] = 'gray' # 设置灰度输出

# 在ipython（jupyter基于ipython）里已经import过的模块修改后需要重新reload的部分自动reload
%load_ext autoreload
%autoreload 2

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

data = get_CIFAR10_data()
for k, v in list(data.items()):
  print(('%s: ' % k, v.shape))
```

### Affine layer: foward

完成cs231n/layers.py 里的affine_forward 方程

