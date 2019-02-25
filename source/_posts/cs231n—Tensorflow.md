---
title: CS231n_Assignment2———Tensorflow
date: 2018-12-13 16:35:19
tags: [CV, cs231n, assignment]
---

这是第二次大作业的倒数第二个，关于tensorflow入门的讲解，自己也买了本相应的书，配套着这几天将tensorlfow和pytorch一同了解下

<!--more-->

## Part I: Preparation

首先是对数据的处理，对cifar10的数据进行归一化(?为什么呢)

```python
def load_cifar10(num_training=49000, num_validation=1000, num_test=10000):
    """
    Fetch the CIFAR-10 dataset from the web and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 dataset and use appropriate data types and shapes
    cifar10 = tf.keras.datasets.cifar10.load_data()
    (X_train, y_train), (X_test, y_test) = cifar10
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32).flatten()
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32).flatten()

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean pixel and divide by std
    mean_pixel = X_train.mean(axis=(0, 1, 2), keepdims=True)
    std_pixel = X_train.std(axis=(0, 1, 2), keepdims=True)
    X_train = (X_train - mean_pixel) / std_pixel
    X_val = (X_val - mean_pixel) / std_pixel
    X_test = (X_test - mean_pixel) / std_pixel

    return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
NHW = (0, 1, 2)
X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape, y_train.dtype)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
```

### Preparation: Dataset object

1、得到数据集后需要将所有的数据构建成一个Dataset的类，目的是能够一起迭代datas和labels，这一点的意识我一直没有，看到了更复杂更大的代码后发现别人都采取了这样的方式，值得借鉴

2、设置硬件的设备号，看用那个cpu,哪个gpu

> For our own convenience we'll define a lightweight `Dataset` class which lets us iterate over data and labels. 

```python
class Dataset(object):
    def __init__(self, X, y, batch_size, shuffle=False):
        """
        Construct a Dataset object to iterate over data X and labels y
        
        Inputs:
        - X: Numpy array of data, of any shape
        - y: Numpy array of labels, of any shape but with y.shape[0] == X.shape[0]
        - batch_size: Integer giving number of elements per minibatch
        - shuffle: (optional) Boolean, whether to shuffle the data on each epoch
        """
        assert X.shape[0] == y.shape[0], 'Got different numbers of data and labels'
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        # iter()是一个迭代器
        return iter((self.X[i:i+B], self.y[i:i+B]) for i in range(0, N, B))


train_dset = Dataset(X_train, y_train, batch_size=64, shuffle=True)
val_dset = Dataset(X_val, y_val, batch_size=64, shuffle=False)
test_dset = Dataset(X_test, y_test, batch_size=64)
# We can iterate through a dataset like this:
for t, (x, y) in enumerate(train_dset):
    print(t, x.shape, y.shape)
    if t > 5: break

# Set up some global variables
USE_GPU = False

if USE_GPU:
    device = '/device:GPU:0'
else:
    device = '/cpu:0'

# Constant to control how often we print when training models
print_every = 100

print('Using device: ', device)
```

## Part II: Barebone TensorFlow

接下来是将tensorflow的运行机制，tensorflow本质上是静态图，而pytorch是动态图，所以需要分为两步骤:

- 定义静态图：
  - 重设置图，tf.reset_default_graph()
  - 确定运行设备，with tf.device(device)
  - 定义 `placeholder` （从外部输入到图中的数据）+存储在图中需要更新的变量（w）和操作(卷积)
- 运行
  - 开启会话，with tf.Session() as sess
  - 初始化所有变量，sess.run(tf.global_variables_initializer())
  - 运行，scores_np = sess.run(scores, feed_dict={x: x_np})，feed_dict里面存储的是要输入的`placeholder`



> This means that a typical TensorFlow program is written in two distinct phases:
>
> 1. Build a computational graph that describes the computation that you want to perform. This stage doesn't actually perform any computation; it just builds up a symbolic representation of your computation. This stage will typically define one or more `placeholder` objects that represent inputs to the computational graph.
> 2. Run the computational graph many times. Each time the graph is run you will specify which parts of the graph you want to compute, and pass a `feed_dict`dictionary that will give concrete values to any `placeholder`s in the graph.

```python
def three_layer_convnet(x, params):
    """
    A three-layer convolutional network with the architecture described above.
    
    Inputs:
    - x: A TensorFlow Tensor of shape (N, H, W, 3) giving a minibatch of images
    - params: A list of TensorFlow Tensors giving the weights and biases for the
      network; should contain the following:
      - conv_w1: TensorFlow Tensor of shape (KH1, KW1, 3, channel_1) giving
        weights for the first convolutional layer.
      - conv_b1: TensorFlow Tensor of shape (channel_1,) giving biases for the
        first convolutional layer.
      - conv_w2: TensorFlow Tensor of shape (KH2, KW2, channel_1, channel_2)
        giving weights for the second convolutional layer
      - conv_b2: TensorFlow Tensor of shape (channel_2,) giving biases for the
        second convolutional layer.
      - fc_w: TensorFlow Tensor giving weights for the fully-connected layer.
        Can you figure out what the shape should be?
      - fc_b: TensorFlow Tensor giving biases for the fully-connected layer.
        Can you figure out what the shape should be?
    """
    conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b = params
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer ConvNet.            #
    ############################################################################
    a1 = tf.nn.conv2d(x, conv_w1, strides = [1,1,1,1], padding = 'SAME') + conv_b1
    h1 = tf.nn.relu(a1)
    a2 = tf.nn.conv2d(h1, conv_w2, strides = [1,1,1,1], padding = 'SAME') + conv_b2
    h2 = tf.nn.relu(a2)
    h2_flat = flatten(h2)
    scores = tf.matmul(h2_flat, fc_w) + fc_b
    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
    return scores
```

- tensorflow的默认的输入数据的维度是（N,H,W,C),而pytorch呢是(N,C,H,W)

  > **NOTE**: TensorFlow and PyTorch differ on the default Tensor layout; TensorFlow uses N x H x W x C but PyTorch uses N x C x H x W.

- tf.nn.conv2d()的用法:https://www.tensorflow.org/api_docs/python/tf/nn/conv2d

  > ```python
  > tf.nn.conv2d(
  >     input,
  >     filter,
  >     strides,
  >     padding,
  >     use_cudnn_on_gpu=True,
  >     data_format='NHWC',
  >     dilations=[1, 1, 1, 1],
  >     name=None
  > )
  > # input()： (N, H, W, 3)
  > # filter()：(KH1, KW1, 3, channel_1）
  > # strides: 必须有strides[0] = strides[3] = 1
  > # padding: "SAME", "VALID"
  > # VALID是采用丢弃的方式,比如上述的input_width=13,只允许滑动2次,多余的元素全部丢掉
  > # SAME的方式,采用的是补全的方式,对于上述的情况,允许滑动3次,但是需要补3个元素,左奇右偶,在左边补一个0,右边补2个0
  > # NHWC：[batch, in_height, in_width, in_channels]
  > # NCHW：[batch, in_channels, in_height, in_width]
  > ```


```python
def three_layer_convnet_test():
    tf.reset_default_graph()

    with tf.device(device):
        x = tf.placeholder(tf.float32)
        conv_w1 = tf.zeros((5, 5, 3, 6))
        conv_b1 = tf.zeros((6,))
        conv_w2 = tf.zeros((3, 3, 6, 9))
        conv_b2 = tf.zeros((9,))
        fc_w = tf.zeros((32 * 32 * 9, 10))
        fc_b = tf.zeros((10,))
        params = [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]
        scores = three_layer_convnet(x, params)

    # Inputs to convolutional layers are 4-dimensional arrays with shape
    # [batch_size, height, width, channels]
    x_np = np.zeros((64, 32, 32, 3))
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        scores_np = sess.run(scores, feed_dict={x: x_np})
        print('scores_np has shape: ', scores_np.shape)

with tf.device('/cpu:0'):
    three_layer_convnet_test()
```

### Barebones TensorFlow: Training Step

- 一个重要的点，就是tensorflow中loss的输出只会根据计算图中的部分节点进行计算，不用依赖权重更新，因此如果按照上面那么写的话无法达到训练的效果

>There is an important bit of subtlety here - when we call sess.run, TensorFlow does not execute all operations in the computational graph; it only executes the minimal subset of the graph necessary to compute the outputs that we ask TensorFlow to produce. 

- 因此为了解决这个问题，在图中插入了一个控制流，强制性先进行权重更新的操作，再进行loss的计算

>To fix this problem, we insert a control dependency into the graph, adding a duplicate loss node to the graph that does depend on the outputs of the weight update operations; this is the object that we actually return from the training_step function.

- 需要留意的是，因为计算图中每一步都需要是op，常规的复制等操作是无法加入控制流的，所以在这里用到了`tf.identity(loss)`，具体的参考链接如下：https://blog.csdn.net/hu_guan_jie/article/details/78495297

- assign(ref, value)/assign_add(ref, value)/assign_sub(ref, value)

  这是个赋值函数，构建于图创建阶段，只有被run了，ref的值才发生变化，后面的两个捏就是加和减完再赋值

- tf.nn.sparse_softmax_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, name=None)

  其中labels是长度为batch的一维向量，logits维度(batch,num_class)，返回值要和lables相同的shape以及logits相同的数据类型，

  > 这个函数和tf.nn.softmax_cross_entropy_with_logits函数比较明显的区别在于它的参数labels的不同，这里的参数label是非稀疏表示的，比如表示一个3分类的一个样本的标签，稀疏表示的形式为[0,0,1]这个表示这个样本为第3个分类，而非稀疏表示就表示为2（因为从0开始算，0,1,2,就能表示三类），同理[0,1,0]就表示样本属于第二个分类，而其非稀疏表示为1。
  >
  > tf.nn.sparse_softmax_cross_entropy_with_logits（）比tf.nn.softmax_cross_entropy_with_logits多了一步将labels稀疏化的操作。因为深度学习中，图片一般是用非稀疏的标签的，所以用tf.nn.sparse_softmax_cross_entropy_with_logits（）的频率比tf.nn.softmax_cross_entropy_with_logits高。

- tf.reduce_mean (input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)

  函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，主要用作降维或者计算tensor（图像）的平均值，其中如果没有指定axis，默认对所有的的元素求均值

- `for w, grad_w in zip(params, grad_params):`

  将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。在python3中返回的是一个object，如果需要转化为列表则需要手动list()转换

```python
def training_step(scores, y, params, learning_rate):
    """
    Set up the part of the computational graph which makes a training step.

    Inputs:
    - scores: TensorFlow Tensor of shape (N, C) giving classification scores for
      the model.
    - y: TensorFlow Tensor of shape (N,) giving ground-truth labels for scores;
      y[i] == c means that c is the correct class for scores[i].
    - params: List of TensorFlow Tensors giving the weights of the model
    - learning_rate: Python scalar giving the learning rate to use for gradient
      descent step.
      
    Returns:
    - loss: A TensorFlow Tensor of shape () (scalar) giving the loss for this
      batch of data; evaluating the loss also performs a gradient descent step
      on params (see above).
    """
    # First compute the loss; the first line gives losses for each example in
    # the minibatch, and the second averages the losses acros the batch
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=scores)# (N,)
    loss = tf.reduce_mean(losses)

    # Compute the gradient of the loss with respect to each parameter of the the
    # network. This is a very magical function call: TensorFlow internally
    # traverses the computational graph starting at loss backward to each element
    # of params, and uses backpropagation to figure out how to compute gradients;
    # it then adds new operations to the computational graph which compute the
    # requested gradients, and returns a list of TensorFlow Tensors that will
    # contain the requested gradients when evaluated.
    # 计算梯度
    grad_params = tf.gradients(loss, params)
    
    # Make a gradient descent step on all of the model parameters.
    new_weights = []   
    # 将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。在python3中返回的是一个object，如果需要转化为列表则需要手动list()转换
    for w, grad_w in zip(params, grad_params):
        # tf.assign_sub（ref,value,use_locking=None,name=None）将ref减去value
        new_w = tf.assign_sub(w, learning_rate * grad_w)
        new_weights.append(new_w)

    # Insert a control dependency so that evaluting the loss causes a weight
    # update to happen; see the discussion above.
    with tf.control_dependencies(new_weights):
        # 复制op
        return tf.identity(loss)
```

### training loop

这里主要是对于数据的一个循环读取和操作，之前设计的train_step的操作已经定义好了静态图，所以接下来在sess的阶段需要利用之前设计好的迭代生成器进行反复训练

- x = tf.placeholder(tf.float32, [None, 32, 32, 3])

  这样的设计使得batch的更改成为可能，在实际的设计静态图的过程中，placeholder的某一维度可以用none代替从而使得在sess里面输入的那一维可以为任意维度

```python
def train_part2(model_fn, init_fn, learning_rate):
    """
    Train a model on CIFAR-10.
    
    Inputs:
    - model_fn: A Python function that performs the forward pass of the model
      using TensorFlow; it should have the following signature:
      scores = model_fn(x, params) where x is a TensorFlow Tensor giving a
      minibatch of image data, params is a list of TensorFlow Tensors holding
      the model weights, and scores is a TensorFlow Tensor of shape (N, C)
      giving scores for all elements of x.
    - init_fn: A Python function that initializes the parameters of the model.
      It should have the signature params = init_fn() where params is a list
      of TensorFlow Tensors holding the (randomly initialized) weights of the
      model.
    - learning_rate: Python float giving the learning rate to use for SGD.
    """
    # First clear the default graph
    tf.reset_default_graph()
    is_training = tf.placeholder(tf.bool, name='is_training')
    # Set up the computational graph for performing forward and backward passes,
    # and weight updates.
    with tf.device(device):
        # Set up placeholders for the data and labels
        x = tf.placeholder(tf.float32, [None, 32, 32, 3])
        y = tf.placeholder(tf.int32, [None])
        params = init_fn()           # Initialize the model parameters
        scores = model_fn(x, params) # Forward pass of the model
        loss = training_step(scores, y, params, learning_rate)

    # Now we actually run the graph many times using the training data
    with tf.Session() as sess:
        # Initialize variables that will live in the graph
        sess.run(tf.global_variables_initializer())
        for t, (x_np, y_np) in enumerate(train_dset):
            # Run the graph on a batch of training data; recall that asking
            # TensorFlow to evaluate loss will cause an SGD step to happen.
            feed_dict = {x: x_np, y: y_np}
            loss_np = sess.run(loss, feed_dict=feed_dict)
            
            # Periodically print the loss and check accuracy on the val set
            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss_np))
                check_accuracy(sess, val_dset, x, scores, is_training)
```

### Check accuracy

- 因为train和evalue的阶段是公用一个训练图的，但是在评估的时候不需要计算loss的，因此在计算accuracy的时候需要自己重新计算

- tf.argmax(axis = ): 返回最大值的索引值

```python
def check_accuracy(sess, dset, x, scores, is_training=None):
    """
    Check accuracy on a classification model.
    
    Inputs:
    - sess: A TensorFlow Session that will be used to run the graph
    - dset: A Dataset object on which to check accuracy
    - x: A TensorFlow placeholder Tensor where input images should be fed
    - scores: A TensorFlow Tensor representing the scores output from the
      model; this is the Tensor we will ask TensorFlow to evaluate.
      
    Returns: Nothing, but prints the accuracy of the model
    """
    num_correct, num_samples = 0, 0
    for x_batch, y_batch in dset:
        feed_dict = {x: x_batch, is_training: 0}
        scores_np = sess.run(scores, feed_dict=feed_dict)
        y_pred = scores_np.argmax(axis=1)
        num_samples += x_batch.shape[0]
        num_correct += (y_pred == y_batch).sum()
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))
```

### Initialization

```python
def kaiming_normal(shape):
    if len(shape) == 2:
        fan_in, fan_out = shape[0], shape[1]
    elif len(shape) == 4:
        fan_in, fan_out = np.prod(shape[:3]), shape[3]
    return tf.random_normal(shape) * np.sqrt(2.0 / fan_in)
```

### Train a Two-Layer Network

因为w和b是需要在计算图中不断被更新的，所以需要用新的数据形式表示，就是tf.Variable()

```python
def two_layer_fc_init():
    """
    Initialize the weights of a two-layer network, for use with the
    two_layer_network function defined above.
    
    Inputs: None
    
    Returns: A list of:
    - w1: TensorFlow Variable giving the weights for the first layer
    - w2: TensorFlow Variable giving the weights for the second layer
    """
    hidden_layer_size = 4000
    w1 = tf.Variable(kaiming_normal((3 * 32 * 32, 4000)))
    w2 = tf.Variable(kaiming_normal((4000, 10)))
    return [w1, w2]

learning_rate = 1e-2
train_part2(two_layer_fc, two_layer_fc_init, learning_rate)
```

### Train a three-layer ConvNet

```python
def three_layer_convnet_init():
    """
    Initialize the weights of a Three-Layer ConvNet, for use with the
    three_layer_convnet function defined above.
    
    Inputs: None
    
    Returns a list containing:
    - conv_w1: TensorFlow Variable giving weights for the first conv layer
    - conv_b1: TensorFlow Variable giving biases for the first conv layer
    - conv_w2: TensorFlow Variable giving weights for the second conv layer
    - conv_b2: TensorFlow Variable giving biases for the second conv layer
    - fc_w: TensorFlow Variable giving weights for the fully-connected layer
    - fc_b: TensorFlow Variable giving biases for the fully-connected layer
    """
    params = None
    ############################################################################
    # TODO: Initialize the parameters of the three-layer network.              #
    ############################################################################
    conv_w1 = tf.Variable(kaiming_normal((5, 5, 3, 32)))
    conv_b1 = tf.Variable(kaiming_normal((32,)))
    conv_w2 = tf.Variable(kaiming_normal((3, 3, 32, 16)))
    conv_b2 = tf.Variable(kaiming_normal((16,)))
    fc_w = tf.Variable(kaiming_normal((32 * 32 * 16, 10)))
    fc_b = tf.Variable(kaiming_normal((10,)))
    
    params = [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return params

learning_rate = 3e-3
train_part2(three_layer_convnet, three_layer_convnet_init, learning_rate)
```

## Part III: Keras Model API

keras是一个更加封装化的包，步骤如下：

- 定义一个 `tf.keras.model`的子类

- 在类里面的 `__init__()` 函数里，定义所有需要的层作为类属性, `tf.layers` 提供了很多常见的卷积类, 像全卷积网络 `tf.layers.Dense`  和卷积网络 `tf.layers.Conv2D` 

> **Warning**: Don't forget to call `super().__init__()` as the first line in your initializer!

- 使用 `call()` 函数计算前向传播的值，在  `__init__()`里面定义的类拥有的 `__call__()` 类使得每个定义好的layer能够被当作一个函数对象，同时不要在 `call()`里面定义任何层

- 定义好后 后面就可以将这个子类实例化，就和Part2中的定义的模型函数一样使用就好

### Module API:Two-Layer Network

-   `Initializer` 是一个初始化器对象，初始化层内可学习参数的初始值。其中`tf.variance_scaling_initializer(scale=)`作用类似于凯明正则化

-  `tf.layers.Dense` 是全链接层对象，包含激活函数，比如`activation=tf.nn.relu` 

-  `tf.layers.flatten` 用来替代上面自己写的flatten函数，但是真的没有看懂下面的这段注释，需要询问一下

  >Unfortunately the `flatten` function we defined in Part II is not compatible with the `tf.keras.Model` API; fortunately we can use `tf.layers.flatten` to perform the same operation. The issue with our `flatten` function from Part II has to do with static vs dynamic shapes for Tensors, which is beyond the scope of this notebook; you can read more about the distinction [in the documentation](https://www.tensorflow.org/programmers_guide/faq#tensor_shapes).

- 第二个问题，`scores = model(x)`为什么直接就可以呢？明明没有调用call函数啊？

```python
class TwoLayerFC(tf.keras.Model):
    def __init__(self, hidden_size, num_classes):
        super().__init__()        
        initializer = tf.variance_scaling_initializer(scale=2.0)
        self.fc1 = tf.layers.Dense(hidden_size, activation=tf.nn.relu,
                                   kernel_initializer=initializer)
        self.fc2 = tf.layers.Dense(num_classes,
                                   kernel_initializer=initializer)
    def call(self, x, training=None):
        x = tf.layers.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def test_TwoLayerFC():
    """ A small unit test to exercise the TwoLayerFC model above. """
    tf.reset_default_graph()
    input_size, hidden_size, num_classes = 50, 42, 10

    # As usual in TensorFlow, we first need to define our computational graph.
    # To this end we first construct a TwoLayerFC object, then use it to construct
    # the scores Tensor.
    model = TwoLayerFC(hidden_size, num_classes)
    with tf.device(device):
        x = tf.zeros((64, input_size))
        # 为什么直接就可以呢？明明没有调用call函数啊？
        scores = model(x)

    # Now that our computational graph has been defined we can run the graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        scores_np = sess.run(scores)
        print(scores_np.shape)
        
test_TwoLayerFC()
```

### Funtional API: Two-Layer Networ

上面那个是面向对象的写法，下面的是面向过程的，区分就在于面向过程的函数首字母都是小写，面向过程的首字母都是大写

```python
def two_layer_fc_functional(inputs, hidden_size, num_classes):     
    initializer = tf.variance_scaling_initializer(scale=2.0)
    flattened_inputs = tf.layers.flatten(inputs)
    fc1_output = tf.layers.dense(flattened_inputs, hidden_size, activation=tf.nn.relu,
                                 kernel_initializer=initializer)
    scores = tf.layers.dense(fc1_output, num_classes,
                             kernel_initializer=initializer)
    return scores

def test_two_layer_fc_functional():
    """ A small unit test to exercise the TwoLayerFC model above. """
    tf.reset_default_graph()
    input_size, hidden_size, num_classes = 50, 42, 10

    # As usual in TensorFlow, we first need to define our computational graph.
    # To this end we first construct a two layer network graph by calling the
    # two_layer_network() function. This function constructs the computation
    # graph and outputs the score tensor.
    with tf.device(device):
        x = tf.zeros((64, input_size))
        scores = two_layer_fc_functional(x, hidden_size, num_classes)

    # Now that our computational graph has been defined we can run the graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        scores_np = sess.run(scores)
        print(scores_np.shape)
        
test_two_layer_fc_functional()
```

### Three-Layer ConvNet

模仿上面的内容，自己构造出三层的网络，重点就是熟悉`tf.layers.Conv2D` 和`tf.layers.Dense`各个位置的参数

- `tf.layers.Conv2D` (filters,kernel_size,strides,padding,activation,kernel_initializer)

  其中，`fliters` 是输出的维度，`kernel_size`和`strides`输入的是整数/列表/远祖都可以

- `tf.layers.Dense`(**units**,**activation**,**use_bias**,**kernel_initializer**)

  其中，`units` 是输出的维度

```python
class ThreeLayerConvNet(tf.keras.Model):
    def __init__(self, channel_1, channel_2, num_classes):
        super().__init__()
        ########################################################################
        # TODO: Implement the __init__ method for a three-layer ConvNet. You   #
        # should instantiate layer objects to be used in the forward pass.     #
        ########################################################################
        initializer = tf.variance_scaling_initializer(scale=2.0)
        self.conv_2d1 = tf.layers.Conv2D(activation=tf.nn.relu,
                                         filters=channel_1,
                                         kernel_size=(5,5),
                                         strides=(2,2),
                                         padding= 'same',
                                   kernel_initializer=initializer)
        self.conv_2d2 = tf.layers.Conv2D(activation=tf.nn.relu,
                                         filters=channel_2,
                                         kernel_size=(3,3),
                                         strides=(1,1),
                                         padding= 'same',
                                   kernel_initializer=initializer)
        self.fc1 = tf.layers.Dense(num_classes, kernel_initializer=initializer)
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        
    def call(self, x, training=None):
        scores = None
        ########################################################################
        # TODO: Implement the forward pass for a three-layer ConvNet. You      #
        # should use the layer objects defined in the __init__ method.         #
        ########################################################################
        scores = self.conv_2d1(x)
        scores = self.conv_2d2(scores)
        scores = tf.layers.flatten(scores)
        scores = self.fc1(scores)
        
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################        
        return scores
```

### Training Loop

- 这里是利用optimizer这样一个优化器去替代自己写的梯度下降(training_step部分)，然后依旧利用流控制，在更新`train_op`之前先更新图中的UPDATE_OPS操作
-  `Optimizer` 对象可以从 `tf.train` 中获得

```python
# 这里的optimizer是一个函数的返回值，而返回的是GradientDescentOptimizer, AdagradOptimizer, MomentumOptimizer这类的东西
optimizer = optimizer_init_fn()
# tf.GraphKeys.UPDATE_OPS:holds the operators that update the states of the network
# tensorflow的collection提供一个全局的存储机制，不会受到变量名生存空间的影响。一处保存，到处可取
# tf.get_collection:从collection中获取数据
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#optimizer.minimize：An Operation that updates the variables in var_list. 
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss)
```
```python
def train_part34(model_init_fn, optimizer_init_fn, num_epochs=1):
    """
    Simple training loop for use with models defined using tf.keras. It trains
    a model for one epoch on the CIFAR-10 training set and periodically checks
    accuracy on the CIFAR-10 validation set.
    
    Inputs:
    - model_init_fn: A function that takes no parameters; when called it
      constructs the model we want to train: model = model_init_fn()
    - optimizer_init_fn: A function which takes no parameters; when called it
      constructs the Optimizer object we will use to optimize the model:
      optimizer = optimizer_init_fn()
    - num_epochs: The number of epochs to train for
    
    Returns: Nothing, but prints progress during trainingn
    """
    tf.reset_default_graph()    
    with tf.device(device):
        # Construct the computational graph we will use to train the model. We
        # use the model_init_fn to construct the model, declare placeholders for
        # the data and labels
        x = tf.placeholder(tf.float32, [None, 32, 32, 3])
        y = tf.placeholder(tf.int32, [None])
        
        # We need a place holder to explicitly specify if the model is in the training
        # phase or not. This is because a number of layers behaves differently in
        # training and in testing, e.g., dropout and batch normalization.
        # We pass this variable to the computation graph through feed_dict as shown below.
        is_training = tf.placeholder(tf.bool, name='is_training')
        
        # Use the model function to build the forward pass.
        scores = model_init_fn(x, is_training)

        # Compute the loss like we did in Part II
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=scores)
        loss = tf.reduce_mean(loss)

        # Use the optimizer_fn to construct an Optimizer, then use the optimizer
        # to set up the training step. Asking TensorFlow to evaluate the
        # train_op returned by optimizer.minimize(loss) will cause us to make a
        # single update step using the current minibatch of data.
        
        # Note that we use tf.control_dependencies to force the model to run
        # the tf.GraphKeys.UPDATE_OPS at each training step. tf.GraphKeys.UPDATE_OPS
        # holds the operators that update the states of the network.
        # For example, the tf.layers.batch_normalization function adds the running mean
        # and variance update operators to tf.GraphKeys.UPDATE_OPS.
        optimizer = optimizer_init_fn()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)

    # Now we can run the computational graph many times to train the model.
    # When we call sess.run we ask it to evaluate train_op, which causes the
    # model to update.
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        t = 0
        for epoch in range(num_epochs):
            print('Starting epoch %d' % epoch)
            for x_np, y_np in train_dset:
                feed_dict = {x: x_np, y: y_np, is_training:1}
                loss_np, _ = sess.run([loss, train_op], feed_dict=feed_dict)
                if t % print_every == 0:
                    print('Iteration %d, loss = %.4f' % (t, loss_np))
                    check_accuracy(sess, val_dset, x, scores, is_training=is_training)
                    print()
                t += 1
```

### Train a Two-Layer Network

`GradientDescentOptimizer`（learning_rate,use_locking,name）

- **learning_rate**: A Tensor or a floating point value. The learning rate to use.
- **use_locking**: If True use locks for update operations.
- **name**: Optional name prefix for the operations created when applying gradients. Defaults to "GradientDescent".

```python
hidden_size, num_classes = 4000, 10
learning_rate = 1e-2

def model_init_fn(inputs, is_training):
    return TwoLayerFC(hidden_size, num_classes)(inputs)

def optimizer_init_fn():
    return tf.train.GradientDescentOptimizer(learning_rate)

train_part34(model_init_fn, optimizer_init_fn)
```

### Train a Two-Layer Network (functional API)

```python
hidden_size, num_classes = 4000, 10
learning_rate = 1e-2

def model_init_fn(inputs, is_training):
    return two_layer_fc_functional(inputs, hidden_size, num_classes)

def optimizer_init_fn():
    return tf.train.GradientDescentOptimizer(learning_rate)

train_part34(model_init_fn, optimizer_init_fn)
```

### Train a Three-Layer ConvNet

```python
learning_rate = 3e-3
channel_1, channel_2, num_classes = 32, 16, 10

def model_init_fn(inputs, is_training):
    model = None
    
    ############################################################################
    # TODO: Complete the implementation of model_fn.                           #
    ############################################################################
    model = ThreeLayerConvNet(channel_1, channel_2, num_classes)
    ############################################################################
    #                           END OF YOUR CODE                               #
    ############################################################################
    return model(inputs)

def optimizer_init_fn():
    optimizer = None
    ############################################################################
    # TODO: Complete the implementation of model_fn.                           #
    ############################################################################
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    ############################################################################
    #                           END OF YOUR CODE                               #
    ############################################################################
    return optimizer

train_part34(model_init_fn, optimizer_init_fn)
```

## Part IV: Keras Sequential API

很多时候有时候不需要那么复杂的写出自定义的类，往往是前一个类的输出是后一个类的输入，所以出现了`tf.keras.Sequential`

### Two-Layer Network

```python
learning_rate = 1e-2

def model_init_fn(inputs, is_training):
    input_shape = (32, 32, 3)
    hidden_layer_size, num_classes = 4000, 10
    initializer = tf.variance_scaling_initializer(scale=2.0)
    layers = [
        tf.layers.Flatten(input_shape=input_shape),
        tf.layers.Dense(hidden_layer_size, activation=tf.nn.relu,
                        kernel_initializer=initializer),
        tf.layers.Dense(num_classes, kernel_initializer=initializer),
    ]
    model = tf.keras.Sequential(layers)
    return model(inputs)

def optimizer_init_fn():
    return tf.train.GradientDescentOptimizer(learning_rate)

train_part34(model_init_fn, optimizer_init_fn)
```

### Three-Layer ConvNet

```python
def model_init_fn(inputs, is_training):
    model = None
    ###############z#############################################################
    # TODO: Construct a three-layer ConvNet using tf.keras.Sequential.         #
    ############################################################################
    input_shape = (64, 32, 32, 3)
    
    channel_1, channel_2, num_classes = 12, 8, 10
    initializer = tf.variance_scaling_initializer(scale=2.0)
    
    layers = [    
        tf.layers.Conv2D(input_shape=input_shape，
                                         activation=tf.nn.relu,
                                         filters=channel_1,
                                         kernel_size=(5,5),
                                         strides=(2,2),
                                         padding= 'same',
                                         kernel_initializer=initializer)，
        tf.layers.Conv2D(activation=tf.nn.relu,
                                         filters=channel_2,
                                         kernel_size=(3,3),
                                         strides=(1,1),
                                         padding= 'same',
                                         kernel_initializer=initializer)，
        tf.layers.Flatten()，
        tf.layers.Dense(num_classes, kernel_initializer=initializer)     
    ]
    model = tf.keras.Sequential(layers)
    ############################################################################
    #                            END OF YOUR CODE                              #
    ############################################################################
    return model(inputs)

learning_rate = 5e-4
def optimizer_init_fn():
    optimizer = None
    ############################################################################
    # TODO: Complete the implementation of model_fn.                           #
    ############################################################################
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    ############################################################################
    #                           END OF YOUR CODE                               #
    ############################################################################
    return optimizer

train_part34(model_init_fn, optimizer_init_fn)
```



## Part V: CIFAR-10 open-ended challenge

- 接下来是自己去调试各种参数，需要注意的一点是这里的函数无法用BN/Dropout，但为啥捏？？？？？从下面作者的解释来讲，是因为这里代码的training loop的设置问题导致，但是为什么呢？

>#### WARNING: Batch Normalization / Dropout
>
>Batch Normalization and Dropout **WILL NOT WORK CORRECTLY** if you use the `train_part34()`function with the object-oriented `tf.keras.Model` or `tf.keras.Sequential` APIs; if you want to use these layers with this training loop then you **must use the tf.layers functional API**.
>
>We wrote `train_part34()` to explicitly demonstrate how TensorFlow works; however there are some subtleties that make it tough to handle the object-oriented batch normalization layer in a simple training loop. In practice both `tf.keras` and `tf` provide higher-level APIs which handle the training loop for you, such as [keras.fit](https://keras.io/models/sequential/) and [tf.Estimator](https://www.tensorflow.org/programmers_guide/estimators), both of which will properly handle batch normalization when using the object-oriented API.

- 下面是提供的一些建议，我觉得重要的一点是：参数如果设置合适的话，会在前几百步就有效果，所以以后在调参的时候要注意

>#### Tips for training
>
>For each network architecture that you try, you should tune the learning rate and other hyperparameters. When doing this there are a couple important things to keep in mind:
>
>- If the parameters are working well, you should see improvement within a few hundred iterations
>- Remember the coarse-to-fine approach for hyperparameter tuning: start by testing a large range of hyperparameters for just a few training iterations to find the combinations of parameters that are working at all.
>- Once you have found some sets of parameters that seem to work, search more finely around these parameters. You may need to train for more epochs.
>- You should use the validation set for hyperparameter search, and save your test set for evaluating your architecture on the best parameters as selected by the validation set.