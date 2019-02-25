---
title: 笔记——Tensorflow实战Google深度学习框架（三）
date: 2018-12-17 20:44:59
tags: [CV, 笔记, tensorflow]
---

TensorFlow 入门

<!--more-->

## 3.1 TensorFlow 计算模型——计算图

### 3.1.1 计算图的概念

Tensor：张量（也可以理解为多维数组）

Flow：流（tensor之间通过计算相互转化的过程）

Tensorflow：是一个通过计算图来表诉计算的编程系统，其中**每一个计算**都是计算图上的**一个节点**，而**节点之间的边**描述了**计算之间的依赖关系![屏幕快照 2018-12-17 下午9.20.10](https://ws3.sinaimg.cn/large/006tNbRwly1fya23q81vfj30ii062gnt.jpg)**

### 3.1.2 计算图的使用

程序分为两个阶段，**图的构建**和**图的运行**

- 图的构建：

  - 首先程序 自动构建一个默认的计算图，可以用`tf.get_default_graph()`获取

  - 其次定义的每个计算都会自动地转化为计算图上的节点，查看某个运算所属的图用`a.graph`,没有特殊指定的情况下属于当前默认的计算图

  ```python
  >>> import tensorflow as tf
  >>> a = tf.constant([1,2],name="a")
  >>> b = tf.constant([2,3],name="b")
  >>> result = a+b
  >>> a.graph
  <tensorflow.python.framework.ops.Graph object at 0x1018f4ac8>
  >>> tf.get_default_graph()
  <tensorflow.python.framework.ops.Graph object at 0x1018f4ac8>
  >>> tf.get_default_graph() is a.graph
  True
  ```

  - 用`tf.Graph`生成新的计算图，不同计算图上的张量和运算不会共享
    - 要注意的是，最后一个量e不是定义在with语句里面的，也就是说，e会包含在最开始的那个图中。也就是说，要在某个graph里面定义量，要在with语句的范围里面定义

  ```python
  import tensorflow as tf
  import numpy as np
  
  c=tf.constant(value=1)
  print(c.graph is tf.get_default_graph())
  print(c.graph)
  print(tf.get_default_graph())
  #True
  #<tensorflow.python.framework.ops.Graph object at 0x111a55160>
  #<tensorflow.python.framework.ops.Graph object at 0x111a55160>
  
  g1=tf.Graph()
  print("g1:",g1)
  with g1.as_default():
      d=tf.constant(value=2)
      print(d.graph)
  #g1: <tensorflow.python.framework.ops.Graph object at 0x1836b8a828>
  #<tensorflow.python.framework.ops.Graph object at 0x1836b8a828>
  
  g2=tf.Graph()
  print("g2:",g2)
  g2.as_default()
  e=tf.constant(value=15)
  print(e.graph)
  #g2: <tensorflow.python.framework.ops.Graph object at 0x1836b8a748>
  #<tensorflow.python.framework.ops.Graph object at 0x111a55160>
  ```

  - `tf.Graph.device`指定运行计算的设备

  - ```python
    g = tf.Graph()
    with g.device('/gpu:0'):
    	result = a + b
    ```

  - 通过集合（collection）管理计算图中的不同的计算资源

    ```
    # 将资源加入集合
    tf.add_collection
    # 获取集合里的资源
    tf.get_collection
    ```

    ![屏幕快照 2018-12-19 下午4.16.26](https://ws4.sinaimg.cn/large/006tNbRwly1fyc4kblut1j30xg0a4jzh.jpg)

## 3.2 Tensorflow 数据模型——张量

## 3.2.1 张量的概念

- 简单理解为多维数组；但是实现时并不采用数组的形式，而是**对计算结果的引用**；

- 在张量中没有保存数据，而是保存了如何得到数字的计算过程

  ```python
  import tensorflow as tf
  a = tf.constant([1,2],name="a")
  b = tf.constant([2,3],name="b")
  result = a+b
  print(result)
  
  #Tensor("add:0", shape=(2,), dtype=int32)
  ```

- 张量包括：name, shape, types

  - name: “node:src_output” ，唯一标志符

    - 和节点一一对应，src_output 指的是这个节点的第几个输出？？？

  - shape:维度

  - types: 注意参与运算的张量的类型必须一致

    ```python
    # float和int型不一致会出错
    # 将a改成“a= tf.constant([l, 2], name="a”, dtype=tf.float32)”就不会报错，所以一般建议指定张量的
    import tensorflow as tf
    a = tf. constant ([1, 2], name=”a”)
    b = tf.constant([2.0, 3.0], name=”b”) 
    result = a + b
    #ValueError: Tensor conversion requested dtype int32 for Tensor  with dtype float32 :’ Tensor (”b : 。”， shape=(2 ,), dtype=float32 ) ’
    ```

### 3.2.2 张量的使用

张量的使用主要可以总结为两大类。

- 第一类用途是**对中间计算结果的引用**。当一个计算包含很多中间结果时，使用张量可以提高代码的可读性，并且可以保留中间结果以求做一些改动

```python
a = tf.constant([1.,2.],name = 'a')
b = tf.constant([2.,3.],name = 'b')
result = a + b
```

- 比如下面这种的可读性就很差

```pyhton
result = tf.constant([1.,2.],name = 'a') + tf.constant([2.,3.].name = 'b')
```

- 第二类情况是当计算图构造完成之后，张量可以用来**获得计算结果**，也就是得到真实的数字。

```python
print(tf.Session().run(result))
#array([ 3.,  5.], dtype=float32)
```

## 3.3 TensorFlow运行模型——会话

通过TensorFlow中的会话（session）来执行定义好的运算。会话拥有并管理TensorFlow程序运行时的所有资源，当所有计算完成之后需要关闭会话来帮助系统回收资源，否则就可能出现资源泄漏的问题。

有两种会话使用模式。

- 第一种，明确会话生成函数和关闭会话函数，存在一个风险：就是系统异常时没法执行Session().close()函数，会使得资源泄漏，解决的方式就是使用上下文管理器

```python
#创建一个会话
sess=tf.Session()
#使用会话来得到运算结果
sess.run(result)
array([ 3.,  5.], dtype=float32)
#关闭会话使本次运行中使用到的资源可以被释放
sess.close()
```

- 第二种，通过Python的上下文管理器来使用会话。所以计算放在“with”内部，上下文管理器退出的时候会自动释放所有资源。

  ```python
  # 不需要Session().close()去关闭会话
  with tf.Session() as sess:
      sess.run(result)
  ```

- 计算图会在创建时会有一个默认图，会话也有类似的机制，但是默认的会话必须手动指定，当默认的会话被指定后，可以用tf.Tensor.eval函数来计算张量的取值

  ```python
  sess = tf.Session()
  with sess.as_default():
  	print(result.eval())
  ```

  - 等价于

  ```python
  sess = tf.Session()
  print(sess.run(result))
  sess.close()
  ```

  - 等价于

  ```python
  sess = tf.Session()
  print(result.eval(session=sess))
  sess.close()
  ```

  - 在交互的环境下有一个直接构建默认会话的函数

  ```python
  sess = tf.InteractivateSession()
  print(result.eval())
  sess.close()
  ```

- 通过`tf.ConfigProto` 来配置需要生成的会话的：并行的线程数、GPU的分配策略等

  - allow_soft_placement: 为TRUE时，表示**在以下任意一个条件成立时， GPU上的运算可以放到 CPU上进行**，默认为false，但为了代码可移植性，以及为了程序能够在拥有不同gpu的机器中运行，建议设为TRUE![屏幕快照 2018-12-20 下午1.47.57](https://ws4.sinaimg.cn/large/006tNbRwly1fyd5w95cgtj30va03mad0.jpg)
  - log_device_placement:为TRUE时，表示**日志中将会记录每个节点被安排在 哪个设备上以方便调试**，在生产环境中一般设置为FALSE，可以减少日志量

  ```python
  config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
  sessl = tf.InteractiveSession(config=config) 
  sess2 = tf.Session(config=config)
  ```

## 3.4 TensorFlow 实现神经网络

### 3.4.1

讲解了一个网页在线训练的模式，暂时不需要了解

### 3.4.2

`tf.matmul(x,w)`:矩阵的乘法

### 3.4.3神经网络参数和Tensorflow变量

`tf.Variable()`定义一个变量，参数是初始化这个参数的方法，比如随机数，常数等

- 随机数：`weights= tf.Variable(tf.random_normal([2, 3], stddev=2))`![屏幕快照 2018-12-20 下午4.52.18](https://ws1.sinaimg.cn/large/006tNbRwly1fydb8076phj30wy07iafa.jpg)

- 常数：`biases= tf.Variable(tf.zeros([3]))`

  ![屏幕快照 2018-12-20 下午4.53.51](https://ws4.sinaimg.cn/large/006tNbRwly1fydba3c3ffj30xa07gdkw.jpg)

- 通过其他变量的初始值来初始化新的变量

  `w2 = tf.Variable (weights.initialized_value()) `

  `w3 = tf.Variable(weights.initialized_value()*2.0) `

- Tensorflow中变量必须被明确初始化后才可以进行计算![屏幕快照 2018-12-20 下午4.57.46](https://ws1.sinaimg.cn/large/006tNbRwly1fydbdmx1qjj30vg0ma7vo.jpg)

- 因为单个变量单独初始化会很麻烦，所以引入了`tf.global_variables_initializer()`

- 变量可以分为`可训练变量`和`不可训练变量`，通过在训练时申明`trainable`为true还是为false来区分需要优化的参数和其他参数，神经网络中默认的优化算法是针对可训练的变量的
  - 所有的变量都会自动加入`GraphKeys.VARIABLES`集合中，而可训练的变量则会自动加入到`GraphKeys.TRAINABLE_VARJABLES `
  - `tf.trainable_variables`函数：得到所需要优化的参数
- 其余的变量的属性有shape和type，一般张量操作之间的这两个参数都得一致，但是维度是可以不一致的，需要设置参数` validate_shape=False`,但一般不建议![屏幕快照 2018-12-20 下午5.14.08](https://ws2.sinaimg.cn/large/006tNbRwly1fydbuthrvdj30ve07odpd.jpg)

### 3.4.4 通过 TensorFlow 训练神经网络模型

在训练的过程中，需要输入数据，如果把所有的数据都生成常量的话，会非常大，所以引入了`tf.placeholder()`这个函数

- 在定义时，类型必须给出，维度不一定要给出，或者只给出某一维度也可以

`x = tf.placeholder(tf.float32 , shape=(l, 2), name=”input")`

- 在run的时候，就需要提供一个` feed_dict`来指定 `所有placeholder`的取值

`print(sess.run(y , feed_dict={x: [[0.7,0.9]]}))`

- 下面是一个简易的训练过程，计算交叉熵损失，训练，再加上`sess.run(train_step)`,就可以进行一个训练了（`tf.clip_by_value()`是限定输入值的值域范围的函数）

![屏幕快照 2018-12-20 下午5.25.41](https://ws2.sinaimg.cn/large/006tNbRwly1fydc896qwej30vg0bm7j1.jpg)

### 3.4.5 完整神经网络样例程序

下面的程序实现一个简单的二分类问题的训练

注意一个之前一直不太懂的做法，`在 shape 的一个维度上使用 None 可以方便使用不同的 batch 大小`。在训练时需要把数据分成比较小的 batch， 但是在测试时，可以一次性使用全部的数据。当数据集比较小时这样比较方便测试，但数据集比较大时，将大量数据放入一个 batch 吁能会导致内存溢出。
### ![屏幕快照 2018-12-20 下午5.34.04](https://ws3.sinaimg.cn/large/006tNbRwly1fydcg5hsk5j30v40myhdh.jpg)

![屏幕快照 2018-12-20 下午5.34.34](https://ws3.sinaimg.cn/large/006tNbRwly1fydcgtja4hj30u013rkjl.jpg)

![屏幕快照 2018-12-20 下午5.34.43](https://ws2.sinaimg.cn/large/006tNbRwly1fydcgy6hewj30uk0nkkhf.jpg)

