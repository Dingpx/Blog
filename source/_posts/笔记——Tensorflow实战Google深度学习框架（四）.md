---
title: 笔记——Tensorflow实战Google深度学习框架（四）
date: 2018-12-21 16:25:55
tags: [CV, 笔记, tensorflow]
---

## 第4章 深层神经网络

开始对神经网络内部结构的了解.

<!--more-->

## 4.1 深度学习和神经网络

### 4.1.1 线性模型的局限性

- 多层的线性模型累加依旧是线性模型，所以面对非线性可分的问题就会陷入困境

![屏幕快照 2018-12-21 下午4.29.54](/Users/dpx/Desktop/屏幕快照 2018-12-21 下午4.29.54.png)

### 4.1.2 激活函数实现去线性化

- tensorflow提供了七种非线性激活函数，tf.nn.relu、 tf.sigrnoid 和 tf.tanh 是其中比较常用的几个 

```python
a= tf.nn.relu (tf.matmul(x, wl) + biasesl) 
y= tf.nn.relu (tf.matmul(a, w2) + biases2)
```

### 4.1.3 多层网络解决异或运算

- 一开始先提出的mlp感知机，是两层的神经网络，但是它无法解决异或的问题。

- 之后有人加了一个隐藏层成功做到了，这说明了隐藏层具有重新组合特征的能力，也是以后越来越深网络的基础

## 4.2 损失函数定义

### 4.2.1 经典损失函数

- 对于多分类的问题，考虑**交叉熵**

  - 但是交叉熵函数考虑的是**两个概率分布**之间的关系，所以需要在神经网络后面加一层softmax

    - H(p ,q) =-∑p(x)log q(x)
    - 交叉熵不是对称的（ H(p, q)!=H(q,p) ），它刻画的是通过概率分布 q 来表达概率分布 p 的困难程度，p代表的是正确答案， q代表的是预测值。
    - 熵越小，越接近结果

    ```python
    # y_为正确结果；y为预测结果
    # tf.clip_by_value(y, le-10, 1.0)规范了值域，防止有0这个值出现
    # *是矩阵元素间相乘
    # 本来是生成（n,m）维度的矩阵，然后对每一行求和后再对n行使用求平均，但是！因为类别数量（C）是不变的，所以就直接对整个矩阵求和也ok
    cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, le-10, 1.0)))
    ```

    - tf提供了两种工具的结合体：`tf.nn.softmax_cross_entropy_with_logits（labels=y , logits=y）`或者`sparse_softmax_cross_entropy_with_logits(labels=None, logits=None)`，唯一的区别是sparse的labels是int类型（即label结果是1，2这之类），而非sparse的labels是one-hot类

- 对于回归问题，用**均方误差**就好`mse = tf.reduce_mean(tf.square(y_ - y))`

### 4.2.2 自定义损失函数

- 有时候输出的目标会很取决于目标函数，所以呢，需要自己自定义损失函数，下面给出书上给出的一个例子，其中会用上`tf.where`,`tf.greater`这两个函数

  - `tf.greater`的输入是两个张量，此函数会比较这两个输入张量中每一个元素的大小，并返回比较结果
  - `tf.where`函数有三个参数。 第一个为选择条件根据， 当选择条件为 True 时， tf.where 函数会选择第二个参数中的值 ， 否则使用第三个参数中的值。

- ![屏幕快照 2018-12-21 下午6.18.24](https://ws3.sinaimg.cn/large/006tNbRwly1fyejbw2w5yj30ug0d2akq.jpg)

  ```
  loss = tf.reduce_sum(tf.where(tf.greater(v1，v2) ,(vl - v2)*a, (v2 - vl)*b))
  ```

## 4.3 神经网络优化算法

方向传播算法需要梯度下降的方法来解决，但是会遇到两个问题

- 不一定到全局最优解（只有凸函数才可以），但是这是所有最优化算法的通病，我们可以用加动量等方式去缓解。
- 迭代时间过长，因为数据量太大。所以考虑用随机梯度下降来做，每次都要选择一个数据进行更新计算，但是又会存在问题，即某一个数据的损失最小不代表全体数据的损失最小，所以最后才做出了以一个折中的办法，即用一个batch来训练

```
batch size = n
#每次读取一小部分数据作为当前的训练数据来执行反向传播算法。
x = tf.placeholder(tf.float32, shape=(batch_size, 2) ， name=’x-input’) 
Y_ = tf.placeholder(tf.float32, shape=(batch_size, 1) ， name=’y-input’)
#定义神经网络结构和优化算法。
loss = .....
train_step = tf.train.AdamOptimizer(0.001) .minimize(loss}
#训练神经网络。
with tf. Session() as sess :
	#参数初始化。
	for i in range(STEPS):
	#准备 batch size 个训练数据。 一般将所有训练数据随机打乱之后再选取可以得到更好的优化效果。
	current X, current Y = sess.run(train_step, feed_dict={x:current_X , y_: current_Y})
```

## 4.4 神经网络进一步优化

下面是一些更细节的做法

### 4.4.1 学习率 的设置

- 学习率的指数衰减：为了前期能够快速更新，而在后期不再因为过大的学习率而达不到收敛，而tf提供了`tf.train.exponential_decay`,相当于实现了下面代码所示的功能

  ```python
  # decay_steps 通常代表了完整的使用 一遍训练数据 所需要的迭代轮数。这个迭代轮数也就是总训练样本数除以每 一个 batch 中的训练样本数 。
  # 一般来说初始学习率、衰减系数和衰减速度都是根据经验设置的。
  decayed_learning_rate = learning_rate * decay_rate * (global_step /decay_steps)
  ```

  注意呢，有两种权重衰减的方式，一种是阶梯状衰减，一种是连续衰减，由参数`staircase`来决定，当为TRUE时，`global_step /decay_steps `会被转化成整数。

  - 阶梯状：每过完一次所有的数据，则学习率变化一次，那么使得所有数据对于模型训练有着相等的作用

  - 连续：不同的训练数据都有不同的学习率，当学习率很小时，对数据的依赖性就会相应的变小了

    ![屏幕快照 2018-12-21 下午7.19.49](https://ws1.sinaimg.cn/large/006tNbRwly1fyel6qux6dj30t80g4tgo.jpg)

  - ```python
    global step= tf.Variable(O)
    #通过 exponential_decay 函数生成学习率
    #0.1 是初始学习率，100是训练完所有的数据的iterations数(迭代轮数)，0.96是衰减系数
    learning rate= tf.train.exponential_decay(0.1, global step, 100, 0.96, staircase=True)
    #在 minimize 函数中传入 global_step 将自动更新global_step参数
    learning_step = tf.train.GradientDescentOptimizer (learning_rate)\
    .minimize( ... my loss ... , global_step=global_step)
    ```

  - 一般来讲，衰减的速度和最后的能够达到的损失没关系，所以不能根据前面几轮的速度快慢去决定优劣

### 4.4.2 过拟合问题

- 正则化：目的是加入一个 **刻画模型复杂程度的指标**，这里也可以理解为去除某些噪音的影响，使得模型更简洁

- **L1和L2的区别**：**L1往往使得结果变得稀疏**，即有很多的参数为0，但是因为L2正则因为很多数在平方时就会变得很小，比如0.001，所以不会太影响loss，所以不容易为0，而是倾向于让结果比较小，比较平滑，那么由此得到的模型会更简单

- `tf.contrib.layers.12（l1）_regularizer` ，其中lambda为系数

- ```python
  w= tf.Variable(tf.random_normal([2 , 1] , stddev=l , seed=l))
  y = tf.matmul(x, w)
  loss= tf.reduce mean(tf.square(y_ - y)) +
  tf.contrib.layers.12 regularizer (lambda) (w)
  ```

- 可以利用集合的方式来分开**结构化损失**和**正则化损失**部分

  - tf.get_collectio()返回一个列表
  - tf.add_n()函数是实现一个列表的元素的相加。就是输入的对象是一个列表，列表里的元素可以是向量，矩阵，等

![屏幕快照 2018-12-21 下午7.56.09](https://ws1.sinaimg.cn/large/006tNbRwly1fyem5m3u3bj30t005e43w.jpg)

![屏幕快照 2018-12-21 下午7.56.01](https://ws2.sinaimg.cn/large/006tNbRwly1fyem5pbdwmj30sw11ee81.jpg)

### 4.4.3 滑动平均模型（不太懂怎么应用在模型中）

（前提是在采用随机梯度下降法训练算法模型的时候）使用滑动平均值的做法可以在很多应用上提升测试集的表现，实质上是利用decay的机制来减缓参数的变化

> 其实滑动平均模型，主要是通过控制衰减率来控制参数更新前后之间的差距，从而达到减缓参数的变化值（如，参数更新前是5，更新后的值是4，通过滑动平均模型之后，参数的值会在4到5之间）

`ema = tf.train.ExponentialMovingAverage(0.99,step)`:定义一个ema类对象

`ema.apply([v1])`:执行更新操作

`ema.average(v1)`:取更新完的值

```python
import tensorflow as tf
 
if __name__ == "__main__":
    #定义一个变量用于计算滑动平均，变量的初始值为0，变量的类型必须是实数
    v1 = tf.Variable(5,dtype=tf.float32)
    #定义一个迭代轮数的变量，动态控制衰减率,并设置为不可训练
    step = tf.Variable(10,trainable=False)
    #定义一个滑动平均类，初始化衰减率为0.99和衰减率的变量step
    ema = tf.train.ExponentialMovingAverage(0.99,step)
    #定义每次滑动平均所更新的列表
    maintain_average_op = ema.apply([v1])
    #初始化上下文会话
    with tf.Session() as sess:
        #初始化所有变量
        init = tf.initialize_all_variables()
        sess.run(init)
        #更新v1的滑动平均值
        '''
        衰减率为min(0.99,(1+step)/(10+step)=0.1}=0.1
        '''
        sess.run(maintain_average_op)
        #[5.0, 5.0]
        print(sess.run([v1,ema.average(v1)]))
        sess.run(tf.assign(v1,4))
        sess.run(maintain_average_op)
        #[4.0, 4.5500002],5*(11/20) + 4*(9/20)
        print(sess.run([v1, ema.average(v1)]))


```

