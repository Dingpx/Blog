---
title: 笔记——Tensorflow实战Google深度学习框架（七）
date: 2019-01-14 21:12:31
tags: [CV, 笔记, tensorflow]
---

> 7.1 节将介绍如何统一输入数据的格式，使得在之后系统中可以更加方便地处理。来自实际问题的数据往往有很多格式和属性，这一节将介绍的 TFRecord格式可以统一不同的原始数据格式， 井更加有效地管理不同的属性。

> 7.2 节将介绍如何对图像数据进行预处理 。 这一节将列举TensorFlow 支持的图像处理函数，并介绍如何使用这些处理方式来弱化与图像识别无关的因素。复杂的图像处理函数有可能降低 训 练的速度，为了加速数据预处理过程 ， 

> 7.3 节将完整地介绍 TensorFlow 利用队列进行多线程数据预处理流程。在这 一节中将首先介绍TensorFlow 中多线程和队列的概念，这是 TensorFlow 多线程数据预处理的基本组成部分。然后将具体介绍数据预处理流程中的每个部分 ， 并将给出 一个完整的多线程数据预处理流程图和程序框架。 

> 7.4 节介绍了最新的数据集( Dataset) API o 数据集从 Tensorflow 1.3 起成为官方推荐的数据输入框架，它使数据的输入和处理大大简 化 。

<!--more-->

### 7.1 TFRecord输入数据格式

#### 7.1.1 TFRecord的格式介绍

- TensorFlow 提供了 TFRecord 的格式来统一存储数据，TFRecord文件中的数据都是通过 tf.train.Example Protocol Buffer的格式存储的

  ![屏幕快照 2019-01-14 下午9.18.03](https://ws1.sinaimg.cn/large/006tNc79ly1fz6fe64qagj30wy0fkk9d.jpg)

- 从上面可以看出，包含一个从属性名称到取值的字典，属性名称是字符串，取值可以为字符列表、实数列表或整数列表
- TFRecords文件包含了`tf.train.Example` 协议内存块(protocol buffer)(协议内存块包含了字段 `Features`)。我们可以写一段代码获取你的数据， 将数据填入到`Example`协议内存块(protocol buffer)，将协议内存块序列化为一个字符串， 并且通过`tf.python_io.TFRecordWriter` 写入到TFRecords文件。
- 从TFRecords文件中读取数据， 可以使用`tf.TFRecordReader`的`tf.parse_single_example`解析器。这个操作可以将`Example`协议内存块(protocol buffer)解析为张量。

#### 7.1.2 TFRecord 样例程序

下面的程序是将输入数据转化为TFRecord的格式

基本的，一个`Example`中包含`Features`，`Features`里包含`Feature`（这里没s）的字典。最后，`Feature`里包含有一个 `FloatList`， 或者`ByteList`，或者`Int64List`

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# 定义函数转化变量类型。
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# 读取mnist数据。
mnist = input_data.read_data_sets("../../datasets/MNIST_data",dtype=tf.uint8, one_hot=True)
images = mnist.train.images
labels = mnist.train.labels
pixels = images.shape[1]
num_examples = mnist.train.num_examples

# 输出TFRecord文件的地址。
filename = "Records/output.tfrecords"
# 创建一个writer 来写TFRecord文件
writer = tf.python_io.TFRecordWriter(filename)
for index in range(num_examples):
    # 将一个图像矩阵转化为一个字符串
    image_raw = images[index].tostring()
	# 将一个样例转化为 Example Protocol Buffer，并将所有的信息写入这个数据结构
    example = tf.train.Example(features=tf.train.Features(feature={
        'pixels': _int64_feature(pixels),
        'label': _int64_feature(np.argmax(labels[index])),
        'image_raw': _bytes_feature(image_raw)
    }))
    # SerializeToString()序列化为字符串
    # 将一个Example写入TFRecord文件
    writer.write(example.SerializeToString())
writer.close()
```

接下来是读取数据

```python
# 读取文件。
reader = tf.TFRecordReader()
# 创建一个队列来维护输入文件列表
filename_queue = tf.train.string_input_producer(["Records/output.tfrecords"])
# 从文件中读出一个样例。也可以使用read_up_to函数一次性读取多个样例。
_,serialized_example = reader.read(filename_queue)

# 解析读取的一个样例。
features = tf.parse_single_example(
    serialized_example,
    features={
        # TensorFlow 提供两种不同的属性解析方法。 一种是方法是 	tf.FixedLenFeature,
        #这种方法解析的结果为一个 Tensor。另一种方法是 tf.VarLenFeature，这种方法 
        #得到的解析结果为 SparseTensor，用于处理稀疏数据。这里解析数据的格式需要和 
        #上面程序写入数据的格式一致。
        'image_raw':tf.FixedLenFeature([],tf.string),
        'pixels':tf.FixedLenFeature([],tf.int64),
        'label':tf.FixedLenFeature([],tf.int64)
    })
# tf.decode_raw可以将字符串解析成图像对应的像索数组。
# tf.cast 是类型转换
images = tf.decode_raw(features['image_raw'],tf.uint8)
labels = tf.cast(features['label'],tf.int32)
pixels = tf.cast(features['pixels'],tf.int32)
sess = tf.Session()

# 启动多线程处理输入数据。
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)
# 每次运行可以读取 TFRecord 文件中的一个样例，
# 当所有样例都读完之后，在此样例中程序会再次读取？？这个地方有些问题的哦
for i in range(10):
    image, label, pixel = sess.run([images, labels, pixels])
```

> 要注意，tensorflow中的队列和普通的队列差不多，不过它里面的`operation`和`tensor`都是符号型的（`symbolic`），在调用`sess.run()`时才执行。

### 7.2 图像数据处理

