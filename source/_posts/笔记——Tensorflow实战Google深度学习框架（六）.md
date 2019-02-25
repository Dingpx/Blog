---
title: 笔记——Tensorflow实战Google深度学习框架（六）
date: 2019-01-14 15:24:33
tags: [CV, 笔记, tensorflow]
---

期末考试结束，开始继续恶补tensorflow，上几章节对于其基本的数据结构有了基本的认识，这一章着重学习在图像上的用处。

>6.1 节将介绍图像识别领域解决的问题以及图像识别领域中经典的数据集 
>
>6.2 节将介绍卷积神经网络的主体思想和整体架构 。 
>
>6.3 节将详细讲解卷积层和池化层的网络结构，以及 TensorFlow 对这些网络结构的支持 
>
>6.4 节中将通过两个 经典的卷积神经网络模型来介绍如何设计卷积神经网络的架构以及如何设置每 一层 神经网络的配置 。这一节将通过 TensorFlow 实现 LeNet-5 模型，并介绍 TensorFlow-Slim 来实现更加复杂的Inception-v3 模型中的 Inception 模块 。
>
>6.5 节中将介绍如何通过 TensorFlow 实现卷积神经网络的迁移学习。

<!--more-->

### 6.1 图像识别问题简介以及经典数据集

- mnist, cifar, imagenet

### 6.2 卷积神经网络

- 问题：全联接网络的参数太过庞大，容易过拟合
- 卷积的优点：局部链接、权值共享
- 一般结构：如下图

![屏幕快照 2019-01-14 下午3.37.37](https://ws4.sinaimg.cn/large/006tNc79ly1fz65jxblsuj30wm0cagt6.jpg)

### 6.3 卷积神经网络常用结构

#### 6.3.1 卷积层

具体的细节就不赘述，细看代码

```python
# 通过 tf.get variable 创建filter的weights和biases
# 前两个维度表示filter的尺寸；第三个表示当前层的深度；第四个表示filter的深度
filter_weight = tf.get_variable (’weights’,[5, 5, 3, 16], initializer=tf.truncated normal,initializer(stddev=0.1))
# 表示filter的深度
biases = tf.get_variable(’biases’,[16],initializer=tf.constant_initializer(0.1))

#输入层是四维矩阵，比如input[O, :, :, :]表示第一张图片
#第二个参数提供了卷积层的权主，
#第三个参数为不同维度上的步长。虽然第三个参数提供的是一个长度为 4 的数组，但是第一维和最后一维的数字要求一定是 l。这是因为卷积层的步氏只对矩阵的长和宽有效
#最后一个参数是填充(padding) 的方法， TensorFlow 中提供 SAME 或是 VALID 两种选择。其中 SAME但表示添加全 0 填充，“VALID”表示不添加
conv = tf.nn.conv2d(
input, filter_weight, strides=[l, 1, 1, l], padding=’SAME’)
# tf.nn.bias_add提供了一个厅便的函数给每一个节点加上偏置顷。
bias= tf.nn.bias_add(conv, biases) 
#将计算结果迦过 ReLU 激活函数完成非线性化。 
actived_conv = tf.nn.relu(bias)
```

#### 6.3.2 池化层

- 使用池化层既可以加快计算速度也有防止过拟合问题的作用
- 卷积层和池化层中过滤器移动的方式是相似的，唯 一 的区别在于
  - 卷积层使用的过滤器是横跨整个深度的，
  - 而池化层使用 的过滤器只影响一个深度上的节点。
- 所以池化层的过滤器除了在长和宽两个维度移动 ，它还需要在深度这个维度移动。

```python
# tf.nn.max_pool 实现了耻火池化层的前向传播过程，它的参数和 tf.nn.conv2d 函数类似。 
# ksize 提供了过滤器的尺寸, strides 提供了步长信息， padding 提供了是否使用全 0 填充。 

pool= tf.nn.max_pool(actived_conv, ksize=[l, 3 , 3, l],
strides=[l, 2, 2, 1], padding=’SAME’)

#tf.nn.avg_pool 来实现平均池化层
```

### 6.4 经典卷积网络模型

> 在 6.4.1 节中将具体介绍 LeNet-5 模型，并给出 一个完整的 TensorFlow 程序来实现 LeNet-5 模型 。通过这个模型，将给出卷积神经网络结构设计的一个通用模式。

> 然后 6.4.2节将介绍设计卷积神经网络
> 结构的另外一种思路一-Inception模型。这个小节将简单介TensorFlow-Slim工具，并通过这个工具实现谷歌提出的 Inception-v3 模型中的 一个模块。

#### 6.4.1 Lenet-5模型

这是inference的部分

```python
import tensorflow as tf
INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

CONV1_DEEP = 32
CONV1_SIZE = 5

CONV2_DEEP = 64
CONV2_SIZE = 5

FC_SIZE = 512
def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")

    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable(
            "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        pool_shape = pool2.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit
```

这是train的部分

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import LeNet5_infernece
import os
import numpy as np

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99

def train(mnist):
    # 定义输出为4维矩阵的placeholder
    x = tf.placeholder(tf.float32, [
            BATCH_SIZE,
            LeNet5_infernece.IMAGE_SIZE,
            LeNet5_infernece.IMAGE_SIZE,
            LeNet5_infernece.NUM_CHANNELS],
        name='x-input')
    y_ = tf.placeholder(tf.float32, [None, LeNet5_infernece.OUTPUT_NODE], name='y-input')
    
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = LeNet5_infernece.inference(x,False,regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y, tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
        
    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            reshaped_xs = np.reshape(xs, (
                BATCH_SIZE,
                LeNet5_infernece.IMAGE_SIZE,
                LeNet5_infernece.IMAGE_SIZE,
                LeNet5_infernece.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})

            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                
def main(argv=None):
    mnist = input_data.read_data_sets("../../../datasets/MNIST_data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    main()
```

#### 6.4.2 Inception-v3 模型

> Inception 模块会首先使用不同尺寸的过滤器处理输入矩阵。
>
>最上方矩阵为使用了边长为 l 的过滤器的卷积层前向传播的结果。类似的,中间矩阵使用的过滤器边长为 3，下方矩阵使用的过滤器边长为 5。
>
>不同的矩阵代表了 Inception模块中的一条计算路径。虽然过滤器的大小不同，但如果所有的过滤器都使用全 0 填充且步长为 l，那么前向传播得到的结果矩阵的长和宽都与输入矩阵一致。
>
>这样经过不同过滤器处理的结果矩阵可以拼接成一个更深的矩阵，可以将它们在深度这个维度上组合起来。

![屏幕快照 2019-01-14 下午4.42.13](https://ws3.sinaimg.cn/large/006tNc79ly1fz67f8wcs8j30jo0eajxr.jpg)

上图中展示的是 Inception 模块的核心思想， 真正在 Inception-v3 模型中使用的 Inception 棋块要更加复杂且多样

![屏幕快照 2019-01-14 下午4.45.32](https://ws3.sinaimg.cn/large/006tNc79ly1fz67ijejx7j30w00ccdmq.jpg)

因为有很多层，如果按照正常的创建卷积层来做的话，需要写很多行代码，这里利用slim这个包来创建。

```python
# 直接使用API实现卷积层
with tf.variable scope(scope name) :
	weights = tf. get_variable (”weight”,...)
	biases= tf .get_variable (”bias”, ...)
	conv = tf.nn.conv2d(...)
	relu = tf.nn.relu(tf.nn.bias add(conv, biases))
    
# 使用 TensorFlow-Slim 实现卷积层。通过 TensorFlow-Slim 可以在一行中实现一个卷积层的前向传播算法。 
# slim.conv2d 函数的有 3 个参数是必填的。
# 第一个参数为输入节点矩阵
# 第二参数是当前卷积层过滤器的深度
# 第三个参数是过滤器的尺寸
# 可边的参数有过滤然移动的步长、是否使用全0填充、激活函数的选择以及变量的命名空间等。 

net = slim.conv2d(Input, 32, [3, 3])
```

下面实现其中的某一块的Inception

```python
slim = tf.contrib.slim
# slim.arg_scope 函数可以用于设置默认的参数取值。 
# 第一个参数是一个函数列表，在这个列表中的函数将使用默认的参数取值
# 后面的参数就是默认的参数取值
with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.arg_pool2d],stride =1, psdding='VALID'):
    ...
    net = 上一层输出节点矩阵
    with tf.variable_scope('Mixed_7c'):
        with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net , 320 , [1 , 1], scope= 'Conv2d_Oa_lxl')
        with tf.variable_scope ('Branch_l'):
            branch_1 = slim.conv2d(net , 384 , [1 , 1], scope= 'Conv2d_Oa_lxl')
            # tf.concat的第一个参数是指定拼接的维度
            branch_1 = tf.concat(3 , [
				slim.conv2d(branch_1, 384, [1,3], scope='Conv2d_Ob_lx3')， 
            	slim.conv2d(branch_1, 384, [3,1], scope='Conv2d_Oc_3xl')
                           ])
		...
        net = tf.concat (3, [branch 0, branch 1])
                                  
```

### 6.5 迁移学习

暂时先放在这里，后期再来补这块