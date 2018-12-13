---
title: CS231n_Assignment2——Dropout
date: 2018-12-10 17:50:27
tags: [CV, cs231n, assignment]
---

完成了bn的部分，开始着手dropout的部分，dropout在本质上就是一种正则化，去使得某些神经元失效来达到防止过拟合的效果

<!--more-->

- 这里的内容比较少，就不完整写出代码了，主要是这里用的是**inverted dropout**，意思就是说，本来在训练阶段需要确定一个要抛弃的概率阈值，使得概率小于此的要抛弃，可是为了保持数据在训练和测试阶段的一致性，测试阶段也需要乘以这个阈值才可以，所以inverted的做法就是在训练阶段就除以这个概率阈值，那么相当于在测试阶段就不需要乘以这个值。

- 下图可以很明显看出在测试阶段的准确率提高了很多

  ![屏幕快照 2018-12-10 下午5.58.53](https://ws1.sinaimg.cn/large/006tNbRwly1fy1sz1k6txj310i0oeacy.jpg)

```python
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = np.random.rand(*x.shape) < (1 - p)
        mask = mask / (1 - p)
        out = mask * x
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
```

