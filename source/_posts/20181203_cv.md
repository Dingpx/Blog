---
layout: posts
title: CS231n Assignment2
date: 2018-12-03 00:35:55
tags: [CV, cs231n, assignment]
---



听完了整体的CNN的框架和常用的调参技巧后，开始着手做assignment2

<!--more-->

[TOC]

以下记录做题时遇到的问题



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

1. tuple类型 不熟悉——用[]取tuple的element
2. 不同维度的矩阵间操作不熟悉——如何扩充还是要找个机会搞清楚

```python
def affine_forward(x, w, b):
"""
Inputs:
- x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
- w: A numpy array of weights, of shape (D, M)
- b: A numpy array of biases, of shape (M,)

Returns a tuple of:
- out: output, of shape (N, M)
- cache: (x, w, b)
"""
out = None
###########################################################################
# TODO: Implement the affine forward pass. Store the result in out. You   #
# will need to reshape the input into rows.                               #
###########################################################################
#print(type(x.shape))   # x.shape的类型是tuple
N = x.shape[0]          # 取tuple类型数组用[]
x = x.reshape(N, -1)    # (N,D)
out = x.dot(w)          # (N,M）
out += b                # (N,M），自动扩充b的维度以适应out

###########################################################################
#                             END OF YOUR CODE                            #
###########################################################################
cache = (x, w, b)
return out, cache
```
在ipynb中测试forward的方程

1. tuple前面加*号，可以变成可变参数传进去，这个是链接，还是没怎么懂（https://zhidao.baidu.com/question/369701615352347164.html）
2. 同时还要搞清楚上面那个和reshape的联合是怎么做的
3. np.prod()：默认计算所有element的乘积，可定义按某个轴计算
4. Np.linspace(): 生成等差数列，默认间隔50，num项自定义间隔

```python
# Test the affine_forward function

num_inputs = 2
input_shape = (4, 5, 6)
output_dim = 3

#Python允许你在list或tuple前面加一个*号，把list或tuple的元素变成可变参数传进去
#print(*input_shape)

input_size = num_inputs * np.prod(input_shape)# np.prod()计算数组元素乘积
weight_size = output_dim * np.prod(input_shape)

# linspace函数默认生成等间隔(50)数列,点明num时以num作为间隔
x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim)
b = np.linspace(-0.3, 0.1, num=output_dim)

out, _ = affine_forward(x, w, b)
correct_out = np.array([[ 1.49834967,  1.70660132,  1.91485297],
                        [ 3.25553199,  3.5141327,   3.77273342]])

# Compare your output with ours. The error should be around e-9 or less.
print('Testing affine_forward function:')
print('difference: ', rel_error(out, correct_out))
```

> Begin Time: 2018/12/03 9:29

### Affine layer: backward

完成cs231n/layers.py 里的affine_forward 方程

1. numpy矩阵的转置：二维数组用**T**属性，高维数组用**transopose**（参考链接：https://www.cnblogs.com/sunshinewang/p/6893503.html)
2. numpy矩阵的乘法：区分好几个名词，点积=内积，适用于向量，而矩阵乘法是适用于矩阵，实质也还是矩阵行列之间的点积，矩阵的元素相乘和这些都没关系；
3. 二维时**np.matmul()**和**np.dot()**等价，而**np.multiply()**和*****实现了元素级别乘法（参考链接：https://blog.csdn.net/u012300744/article/details/80423135)
4. 关于**np.matmul()**和**np.dot()**的区别，要详细看numpy文档（暂时没看），以下链接是别人写的，可以（参考链接https://blog.csdn.net/qq_42698384/article/details/82936294 ）
5. numpy矩阵的sum：记得axis=谁，谁那一轴就消失，比如dout的维度是(10,5),做np.sum(dout,axis=0)，那么结果的维度就是（5，）
6. python默认向量维度是列向量；numpy的一维列向量维度是（n,），二维横向量的维度是（1,n）;一维列向量（如(3,)）在广播运算中是当做二维行向量（如(1,3)）计算的，也就是说(3,)相当于(1,3)
7. 对于db的求导是真滴没搞懂，本质上是张量的乘法吗？这个需要问一下老师🌟有一个参考链接，但是没怎么看懂，http://tieba.baidu.com/p/4139437334
8. 另附上一个矩阵相关求导的公式的链接，以后看(https://blog.csdn.net/max_hope/article/details/80264229)

```python
def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    db = dout # (N,M)
    #print(db.shape)
    db = np.sum(dout,axis=0) # 经常出错记得axis=哪个轴，哪个轴消失就好
    #print(db.shape)
    N = x.shape[0]         # 取tuple类型数组用[]
    x_backvector = x.reshape(N, -1) # (N,D)
    dw = x_backvector.T.dot(dout)# (D,N)*(N,M)=(D,M)
    dx = dout.dot(w.T) #(N,M)*(M,D)=(N,D)
    dx = dx.reshape(x.shape)
    ###########################################################################
    #                             END OF YOUR CODE                            #
  ###########################################################################
    return dx, dw, db
```

在ipynb中测试backward的方程

1. np.random.seed(231),设置种子，为了使后面的随机数按一定的顺序生成，生成随机数的算法没有偏差，参考链接：https://www.cnblogs.com/subic/p/8454025.html
2. lamda函数的写法很有借鉴价值，不知道怎么总结，每一次看到都留意下，争取为自己所用

```python
# Test the affine_backward function
np.random.seed(231)
x = np.random.randn(10, 2, 3)
w = np.random.randn(6, 5)
b = np.random.randn(5)
dout = np.random.randn(10, 5)

dx_num = eval_numerical_gradient_array(lambda x: affine_forward(x, w, b)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: affine_forward(x, w, b)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: affine_forward(x, w, b)[0], b, dout)

_, cache = affine_forward(x, w, b)
dx, dw, db = affine_backward(dout, cache)

# The error should be around e-10 or less
print('Testing affine_backward function:')
print('dx error: ', rel_error(dx_num, dx))
print('dw error: ', rel_error(dw_num, dw))
print('db error: ', rel_error(db_num, db))
```

### ReLU activation: forward+bakward

完成cs231n/layers.py 里的affine_forward 方程

1. 列表推导式：取矩阵小于0的元素并保持维度不变：x[x <= 0] = 0

```python
def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    x[x <= 0] = 0 ## 这种trick要记住，常会用到
    x[x > 0] = 1   ## 一开始写成x[x>=0]= 1, 使得之前的0都变成1了，注意细节
    #print(x)
    dx = dout * x
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx
```

### "Sandwich" layers

为了以后使用的方便，尝试将一些层之间的常用结合结合在一起，比如：affine layer后面经常连着relu layer，就将二者结合在一起变成一个affine_relu_forward 层，这里不用去做，代码已经提供在`cs231n/layer_utils.py`

```python
def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db
```

### Loss layers: Softmax and SVM

这里也是给了以前完成的部分，但是还是值得按照给的标准答案进行分析每一步的trick,首先是SVM

- 列表推导式：

  - `x[np.arange(N), y]`：取input中每个正确类的得分
  - `np.newaxis`在本质上是`None`，在实际使用中往往用做将列向量变成二维横向量

  ```python
  >>> X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
  >>> X[:, 1]
  array([2, 6, 10])       % 这里是一个向量
  >>> X[:, 1].shape       % python默认向量为列向量
  (3, )
  >>>X[:, 1][:, np.newaxis]
  array([[2],
        [6],
        [10]])            % 有时需要返回的矩阵结构即变成（3，1）
  						% 等价于X[:, 1].reshape(-1,1)
  ```

  - `margins > 0`：输出一个等维度的矩阵，满足条件的部分为true，不满足的为false

- `np.zeros_like(x)`: 类似`np.zeros(x.shape)`

- 对svm的求导公式一直不是记忆得很牢哈，简单做个总结：

  - 首先令margin为0的部分为0，不参与求导，即`dx = np.zeros_like(x)`
  - 而margin>0的部分需要考虑两个部分，一个是`xi`,一个是`xy`,即`dx[margins > 0] = 1 # (N,C)`考虑每一个xi求导都是1

  - 而因为`xy`每一次求导时都要得出一个**-1**，所以需要求出所有margin>0的数量num，在最后减去num，即`num_pos = np.sum(margins > 0, axis=1) # (N,)` `dx[np.arange(N), y] -= num_pos  # (N,)-(N,)`

```python
def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]# (N)
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)# (N,C)-(N,1),根据broadcast原则，自动填充为(N,C)
    margins[np.arange(N), y] = 0# 使得真实标签的loss为0，(N,C)
    loss = np.sum(margins) / N# 对所有loss求和并求平均，标量
    num_pos = np.sum(margins > 0, axis=1)# (N,)
    dx = np.zeros_like(x) # (N,C)
    dx[margins > 0] = 1 # (N,C)
    dx[np.arange(N), y] -= num_pos # (N,)-(N,)
    dx /= N
    return loss, dx
```

接着是softmax:

1. `np.max(x, axis=1, keepdims=True)`：不加`keepdims=True`时，维度由(N,C)变成了(N,)，为了保持维度的数目不变，加上后变成了(N,1)。悄悄附上链接，这个总结以后可以常去看，总结numpy维度的问题https://www.jianshu.com/p/2adbf3a44a95
2. softmax的数学推导链接，有机会自己再亲手推导一下

https://blog.csdn.net/yc461515457/article/details/51924604

```python
def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
#为了减少计算量，先进行预处理，那为什么不用mean，我觉得是为了让最大的值在e指数空间里为1，即即使不进行下面的操作，也让所有的数在指数空间里为1
#(N,C)-(N,1)=(N,C)
shifted_logits = x - np.max(x, axis=1, keepdims=True)
#(N,C)变成(N,1)
Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
#计算每个元素的loss的负数，为什么用负的捏，因为需要求下面的每个类的probs,用于下面求梯度用的
log_probs = shifted_logits - np.log(Z)
probs = np.exp(log_probs)
N = x.shape[0]
loss = -np.sum(log_probs[np.arange(N), y]) / N
#这里的求导需要记忆一下，
dx = probs.copy()
dx[np.arange(N), y] -= 1
dx /= N
return loss, dx
```
### Two-layer network

接下里有上面那些的模块化的层了，开始重新构建一个两层的network

- np生成正态分布的函数：`numpy.random.normal(loc=0.0, scale=1.0, size=None)`，参数的意义：

> loc：float
> ​    此概率分布的均值（对应着整个分布的中心centre）
> scale：float
> ​    此概率分布的标准差（对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高）
> size：int or tuple of ints
> ​    输出的shape，默认为None，只输出一个值
> ​    size要不为int要不就是tuple！

- 我们更经常会用到的`np.random.randn(size)`所谓标准正态分布（μ=0,σ=1），对应于`np.random.normal(loc=0, scale=1, size)`
- 给字典对象添加新的对象，例子：`self.params['W1'] = W1`,更多操作见链接：http://www.cnblogs.com/scios/p/8108243.html
- loss函数中值得关注的点：正则化是针对于权重系数的（不包括偏置）？？（有些不太懂）; loss的正则化用的是L2正则，而grads的正则是L1正则

```python
class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        D = input_dim
        H = hidden_dim
        C = num_classes

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        loc = 0.0
        W1_size, b1_size= (D,H), (H,)
        W2_size, b2_size= (H,C), (C,)
        
        W1 = np.random.normal(loc=loc, scale=weight_scale, size=W1_size)
        W2 = np.random.normal(loc=loc, scale=weight_scale, size=W2_size)
        b1 = np.zeros(b1_size)
        b2 = np.zeros(b2_size)
        
        self.params['W1'] = W1
        self.params['W2'] = W2
        self.params['b1'] = b1
        self.params['b2'] = b2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

   def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
   
        affine_relu_out, affine_relu_cache = affine_relu_forward(X, self.params['W1'], self.params['b1'] )
        affine2_out, affine2_cache = affine_forward(affine_relu_out, self.params['W2'], self.params['b2'])
        scores = affine2_out
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)
        #print(self.reg)
        loss += 0.5 * self.reg * np.sum(self.params['W1']  * self.params['W1'] )
        loss += 0.5 * self.reg * np.sum(self.params['W2']  * self.params['W2'] )
        #loss += 0.5 * self.reg * np.sum(self.params['b1']  * self.params['b1'] )
        #loss += 0.5 * self.reg * np.sum(self.params['b1']  * self.params['b1'] )
        #print(loss)
        
        daffine_relu_out, dW2, db2 = affine_backward(dscores, affine2_cache)
        dX, dW1, db1 = affine_relu_backward(daffine_relu_out, affine_relu_cache)
        
        #print(self.reg)
        dW1 += self.reg * self.params['W1']
        dW2 += self.reg * self.params['W2']
        
        grads['W1'] = dW1
        grads['W2'] = dW2
        grads['b1'] = db1
        grads['b2'] = db2
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
```

运行测试时的注意点：

1、`np.random.randin(low,high,size)`:

```python
>>> np.random.randint(5, size=(2, 4))
array([[4, 0, 2, 1],
       [3, 2, 2, 0]])                 
% 生成size个（low，high）中的int数，如果只有一个low/high,则为（0，low/highs）
```

2、`np.random.randn（N，D）`:

生成size = (N，D)范围内的随机数

3、`np.all(b1==0)`:比较两个东西是否相等

4、`np.abs( )`:求绝对值

5、`f = lambda _: model.loss(X, y)[0]`:不知道是不是lambda函数可以不用加入自变量

6、assert函数：assert +条件+报错，用来放在自己的程序中去检验

例子：`assert abs(loss - correct_loss) < 1e-10, 'Problem with training-time loss'`

```python
np.random.seed(231)
N, D, H, C = 3, 5, 50, 7
X = np.random.randn(N, D)
y = np.random.randint(C, size=N)

std = 1e-3
model = TwoLayerNet(input_dim=D, hidden_dim=H, num_classes=C, weight_scale=std)

print('Testing initialization ... ')
W1_std = abs(model.params['W1'].std() - std)
b1 = model.params['b1']
W2_std = abs(model.params['W2'].std() - std)
b2 = model.params['b2']
assert W1_std < std / 10, 'First layer weights do not seem right'
assert np.all(b1 == 0), 'First layer biases do not seem right'
assert W2_std < std / 10, 'Second layer weights do not seem right'
assert np.all(b2 == 0), 'Second layer biases do not seem right'

print('Testing test-time forward pass ... ')
model.params['W1'] = np.linspace(-0.7, 0.3, num=D*H).reshape(D, H)
model.params['b1'] = np.linspace(-0.1, 0.9, num=H)
model.params['W2'] = np.linspace(-0.3, 0.4, num=H*C).reshape(H, C)
model.params['b2'] = np.linspace(-0.9, 0.1, num=C)
X = np.linspace(-5.5, 4.5, num=N*D).reshape(D, N).T
scores = model.loss(X)
correct_scores = np.asarray(
  [[11.53165108,  12.2917344,   13.05181771,  13.81190102,  14.57198434, 15.33206765,  16.09215096],
   [12.05769098,  12.74614105,  13.43459113,  14.1230412,   14.81149128, 15.49994135,  16.18839143],
   [12.58373087,  13.20054771,  13.81736455,  14.43418138,  15.05099822, 15.66781506,  16.2846319 ]])
scores_diff = np.abs(scores - correct_scores).sum()
assert scores_diff < 1e-6, 'Problem with test-time forward pass'

print('Testing training loss (no regularization)')
y = np.asarray([0, 5, 1])
loss, grads = model.loss(X, y)
correct_loss = 3.4702243556
assert abs(loss - correct_loss) < 1e-10, 'Problem with training-time loss'

model.reg = 1.0
loss, grads = model.loss(X, y)
correct_loss = 26.5948426952
assert abs(loss - correct_loss) < 1e-10, 'Problem with regularization loss'

# Errors should be around e-7 or less
for reg in [0.0,0.7]:
  print('Running numeric gradient check with reg = ', reg)
  model.reg = reg
  loss, grads = model.loss(X, y)

  for name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    grad_num = eval_numerical_gradient(f, model.params[name], verbose=False)
    print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))
```

> Finish time 2018/12/03 19:11

### Solver

把训练网络的部分也隔离出网络，形成了一个单独的类solver，这里不用自己编码，看懂之后会去调用这个API去训练自己的网络

