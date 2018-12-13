---
title: CS231n_Assignment2——cnn
date: 2018-12-10 18:11:18
tags: [CV, cs231n, assignment]
---

12月11号，开始写cnn.ipynb的部分

<!--more-->

## Convolution: Naive forward pass

- np.pad()函数https://blog.csdn.net/qq_36332685/article/details/78803622

  **1）语法结构**

  > pad(array, pad_width, mode, **kwargs)
  >
  > 返回值：数组

  **2）参数解释**

  > array——表示需要填充的数组；
  >
  > pad_width——表示每个轴（axis）边缘需要填充的数值数目。 
  > 参数输入方式为：（(before_1, after_1), … (before_N, after_N)），其中(before_1, after_1)表示第1轴两边缘分别填充before_1个和after_1个数值。取值为：{sequence, array_like, int}; (pad,) or int 是before = after 简写形式 
  >
  > mode——表示填充的方式（取值：str字符串或用户提供的函数）,总共有11种填充模式；

  所以如果对 输入维度为(N, C, H, W)的x来卷积的话，要注意第二个参数

  `x_padded = np.pad(x, ((0,), (0,), (pad,), (pad,)), 'constant')`

  这里表示往第三个和第四个维度上pad，比如pad=1的时候，（2，3，4，4）就扩充为（2，3，6，6），注意的是这种写法等价于

  `x_padded = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), 'constant')`

  要不就写全了前后边缘，要不就全省略着写

- 数组的切片范围：

  - 左闭右开

    ```python
    >>> a[0:2]
    array([0., 0.])
    >>> a = [1,2,3,4,5]
    >>> a[0:2]
    [1, 2]
    ```

  - 当某一维的索引确定时，维度将会降低一个维度

    ```
    >>> b = np.ones((1,2,3,4))
    >>> b.shape
    (1, 2, 3, 4)
    >>> a = b[:,1,:,:]
    >>> a
    array([[[1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]]])
    >>> a.shape
    (1, 3, 4)
    ```

  - None 的意思是增加一个维度,等价于numpy.newaxis

    ```python
    >>> a=np.array([[11,12,13,14],[21,22,23,24],[31,32,33,34],[41,42,43,44]])
    >>> print('0维为None:')
    0维为None:
    >>> print(a[None,0:4])
    [[[11 12 13 14]
      [21 22 23 24]
      [31 32 33 34]
      [41 42 43 44]]]
    >>> print('1维为None:')
    1维为None:
    >>> print(a[0:4,None])
    [[[11 12 13 14]]
    
     [[21 22 23 24]]
    
     [[31 32 33 34]]
    
     [[41 42 43 44]]]
    ```

  - 不同维度之间的相加

    ```python
    # out(N*F*H*W),b(F,)
    # 一维和二维的数据相加 ，可以利用broadcast机制自动填充
    # 一维和高维的数据相加呢？
    # 保险的是利用循环，确定相加的那个确定维度
    out[:,f,:,:]+=b[f]
    # 或者将低纬度的数据扩充为高维度
    out = out + b[None, :, None, None]
    ```

  ```python
  def conv_forward_naive(x, w, b, conv_param):
      """
      A naive implementation of the forward pass for a convolutional layer.
  
      The input consists of N data points, each with C channels, height H and
      width W. We convolve each input with F different filters, where each filter
      spans all C channels and has height HH and width WW.
  
      Input:
      - x: Input data of shape (N, C, H, W)
      - w: Filter weights of shape (F, C, HH, WW)
      - b: Biases, of shape (F,)
      - conv_param: A dictionary with the following keys:
        - 'stride': The number of pixels between adjacent receptive fields in the
          horizontal and vertical directions.
        - 'pad': The number of pixels that will be used to zero-pad the input. 
          
  
      During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
      along the height and width axes of the input. Be careful not to modfiy the original
      input x directly.
  
      Returns a tuple of:
      - out: Output data, of shape (N, F, H', W') where H' and W' are given by
        H' = 1 + (H + 2 * pad - HH) / stride
        W' = 1 + (W + 2 * pad - WW) / stride
      - cache: (x, w, b, conv_param)
      """
      out = None
      ###########################################################################
      # TODO: Implement the convolutional forward pass.                         #
      # Hint: you can use the function np.pad for padding.                      #
      ###########################################################################
      pad, stride = conv_param['pad'], conv_param['stride']
      #print(x.shape)
      x_padded = np.pad(x, ((0,), (0,), (pad,), (pad,)), 'constant') # pad alongside four dimensions
      #print(x_padded.shape)
      N, C, H, W = x.shape
      F, C, HH, WW = w.shape
      output_height = 1 + (H + 2 * pad - HH) //stride
      output_width = 1 + (W + 2 * pad - WW) // stride
      #print(type(output_width))
      #print(output_width) 
      out = np.zeros((N, F, output_height, output_width))  
      
      for i in range(output_height):
          for j in range(output_width):
              # (N,C,HH,WW)
              x_padded_mask = x_padded[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
              for k in range(F):
                  # (N,C,HH,WW) * (C,HH,WW) ,axis=(1,2,3)-> (N,1)
                  out[:, k, i, j] = np.sum(x_padded_mask * w[k, :, :, :], axis=(1,2,3))
                      
      out = out + b[None, :, None, None]
      
      ###########################################################################
      #                             END OF YOUR CODE                            #
      ###########################################################################
      cache = (x, w, b, conv_param)
      return out, cache
  
  
  ```


## Aside: Image processing via convolutions

- 切片的方式处理图像

  `d = kitten.shape[1] - kitten.shape[0]`
  `kitten_cropped = kitten[:, d//2:-d//2, :]`

- 手动设计filter的horizontal edges时，这个矩阵的意义不太懂，为啥按照下面那种设置就是水平边缘了呢？

  `w[1, 2, :, :] = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]`

- 在输出图像的时候，为什么只对经过处理的图像做归一化而不对原始图像也做一下呢？

  - 没有做归一化的情况

  ![屏幕快照 2018-12-11 下午6.03.18](https://ws4.sinaimg.cn/large/006tNbRwly1fy2yp3buanj30k20dw7dz.jpg)
  - 做了归一化的情况：

![屏幕快照 2018-12-11 下午5.55.26](https://ws3.sinaimg.cn/large/006tNbRwly1fy2yq6f7boj30m60e8doo.jpg)

```python
from scipy.misc import imread, imresize

kitten, puppy = imread('kitten.jpg'), imread('puppy.jpg')
# kitten is wide, and puppy is already square
# 这里说到kitten的是W>H的，所以采取了切片的手段square这张图片
d = kitten.shape[1] - kitten.shape[0]
kitten_cropped = kitten[:, d//2:-d//2, :]

img_size = 200   # Make this smaller if it runs too slow
x = np.zeros((2, 3, img_size, img_size))
# 将颜色channel的内容放在前面来，符合之前网络的输入的规范
x[0, :, :, :] = imresize(puppy, (img_size, img_size)).transpose((2, 0, 1))
x[1, :, :, :] = imresize(kitten_cropped, (img_size, img_size)).transpose((2, 0, 1))

# Set up a convolutional weights holding 2 filters, each 3x3
w = np.zeros((2, 3, 3, 3))

# The first filter converts the image to grayscale.
# Set up the red, green, and blue channels of the filter.
w[0, 0, :, :] = [[0, 0, 0], [0, 0.3, 0], [0, 0, 0]]
w[0, 1, :, :] = [[0, 0, 0], [0, 0.6, 0], [0, 0, 0]]
w[0, 2, :, :] = [[0, 0, 0], [0, 0.1, 0], [0, 0, 0]]

# Second filter detects horizontal edges in the blue channel.
w[1, 2, :, :] = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

# Vector of biases. We don't need any bias for the grayscale
# filter, but for the edge detection filter we want to add 128
# to each output so that nothing is negative.
b = np.array([0, 128])

# Compute the result of convolving each input in x with each filter in w,
# offsetting by b, and storing the results in out.
out, _ = conv_forward_naive(x, w, b, {'stride': 1, 'pad': 1})

def imshow_noax(img, normalize=True):
    """ Tiny helper to show images as uint8 and remove axis labels """
    if normalize:
        img_max, img_min = np.max(img), np.min(img)
        img = 255.0 * (img - img_min) / (img_max - img_min)
    plt.imshow(img.astype('uint8'))
    # plt.gca()是指获得当前子图
    # plt.gaf()是获取当前图表
    plt.gca().axis('off')

# Show the original images and the results of the conv operation
plt.subplot(2, 3, 1)
imshow_noax(puppy, normalize=False)
plt.title('Original image')
plt.subplot(2, 3, 2)
imshow_noax(out[0, 0])
plt.title('Grayscale')
plt.subplot(2, 3, 3)
imshow_noax(out[0, 1])
plt.title('Edges')
plt.subplot(2, 3, 4)
imshow_noax(kitten_cropped, normalize=False)
plt.subplot(2, 3, 5)
imshow_noax(out[1, 0])
plt.subplot(2, 3, 6)
imshow_noax(out[1, 1])
plt.show()
```

## Convolution: Naive backward pass

- 由于broadcast的机制在高维情况下的扩展会有歧义，所以在这里的计算，都用了大量的循环，不太好直接用向量化的形式表示出来
- 在这里的代码呢，都是先确定好网络各个参数的结构，再去进行相应计算，遵循着结构+算法的结合应用

```python
def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride

    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
    dx = np.zeros_like(x)
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    db += np.sum(dout, axis = (0,2,3))# (F,)

    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
    for i in range(H_out):
        for j in range(W_out):
            #(N,C,HH,WW)
            x_pad_masked = x_pad[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
            for k in range(F): 
                dw[k ,: ,: ,:] += np.sum(x_pad_masked * (dout[:, k, i, j])[:, None, None, None], axis=0)
                for n in range(N): 
                    dx_pad[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] += w[k, :, :, :] *dout[n, k, i, j]
                   
    dx = dx_pad[:,:,pad:-pad,pad:-pad]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


```

## Max-Pooling: Naive forward+Backward

与卷机环节类似，还是需要仔细考虑各个维度之间的对应关系，写出合适的式子

- keepdims的用法

  当它为true的时候，会保持原有的维度的个数，np.max(a,axis=(2,3),keepdims=True)==a这样子的话，就可以容易得出所需要的index,记住依赖的是后者而不是前者

  ```python
  >>> a = np.random.randn(2,2,2,2)
  >>> b = np.max(a,axis=(2,3),keepdims=True)
  >>> b
  array([[[[0.5769296 ]],
  
          [[0.41402278]]],
  
  
         [[[0.67501053]],
  
          [[1.09588594]]]])
  >>> b = np.max(a,axis=(2,3),keepdims=True)==a
  >>> b
  array([[[[False, False],
           [ True, False]],
  
          [[False, False],
           [False,  True]]],
  
  
         [[[ True, False],
           [False, False]],
  
          [[ True, False],
           [False, False]]]])
  ```


```python
def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']
    out_height = 1 + (H - pool_height) //stride
    out_width = 1 + (W - pool_width) // stride
    #out_height = H // pool_height
    #out_width = W // pool_width
    out = np.zeros((N, C, out_height, out_width))
    
    for i in range(out_height):
        for j in range(out_width):
             mask = x[:, :, i*stride:i*stride+pool_height, j*stride:j*stride+pool_width]
             # (N,C,H',W')          
             out[:, :, i, j] = np.max(mask, axis=(2, 3))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']
    dx = np.zeros_like(x)
    out_height = 1 + (H - pool_height) //stride
    out_width = 1 + (W - pool_width) // stride
    for i in range(out_height):
        for j in range(out_width):
          # x, dx has the same dimension, so does x_mask and dx_mask
            x_mask = x[:, :, i*stride:i*stride+pool_height, j*stride:j*stride+pool_width]
            dx_mask = dx[:, :, i*stride:i*stride+pool_height, j*stride:j*stride+pool_width]
            # flags: only the max value is True, others are False
            flags = np.max(x_mask, axis=(2, 3), keepdims=True) == x_mask
            # (N,C,H,W) * (N,C,1,1)
            dx_mask += flags * (dout[:, :, i, j])[:, :, None, None]
                      
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx
```

## Fast layers

- 接下来是一个卷积操作的快速实现版本im2col,本质上是将卷积这种时间上是间断的操作变成时间连续的，这是相关链接（https://blog.csdn.net/dwyane12138/article/details/78449898）![屏幕快照 2018-12-12 下午2.17.43](https://ws3.sinaimg.cn/large/006tNbRwly1fy3xt5d2uwj30co0coq47.jpg)

- 但是note上的这段话还是不太懂：貌似是im2col的方法在pooling上的加速效果并不理想？需要问一下，代码上的注释提供了两种pool的操作方式，一个是reshape(需要tile和square)，一个是im2col的做法

> **NOTE:** The fast implementation for pooling will only perform optimally if the pooling regions are non-overlapping and tile the input. If these conditions are not met then the fast pooling implementation will not be much faster than the naive implementation.

```python
# Rel errors should be around e-9 or less
from cs231n.fast_layers import conv_forward_fast, conv_backward_fast
from time import time
np.random.seed(231)
x = np.random.randn(100, 3, 31, 31)
w = np.random.randn(25, 3, 3, 3)
b = np.random.randn(25,)
dout = np.random.randn(100, 25, 16, 16)
conv_param = {'stride': 2, 'pad': 1}

t0 = time()
out_naive, cache_naive = conv_forward_naive(x, w, b, conv_param)
t1 = time()
out_fast, cache_fast = conv_forward_fast(x, w, b, conv_param)
t2 = time()

print('Testing conv_forward_fast:')
print('Naive: %fs' % (t1 - t0))
print('Fast: %fs' % (t2 - t1))
print('Speedup: %fx' % ((t1 - t0) / (t2 - t1)))
print('Difference: ', rel_error(out_naive, out_fast))

t0 = time()
dx_naive, dw_naive, db_naive = conv_backward_naive(dout, cache_naive)
t1 = time()
dx_fast, dw_fast, db_fast = conv_backward_fast(dout, cache_fast)
t2 = time()

print('\nTesting conv_backward_fast:')
print('Naive: %fs' % (t1 - t0))
print('Fast: %fs' % (t2 - t1))
print('Speedup: %fx' % ((t1 - t0) / (t2 - t1)))
print('dx difference: ', rel_error(dx_naive, dx_fast))
print('dw difference: ', rel_error(dw_naive, dw_fast))
print('db difference: ', rel_error(db_naive, db_fast))
```

```
Testing conv_forward_fast:
Naive: 0.202192s
Fast: 0.009184s
Speedup: 22.015368x
Difference:  4.926407851494105e-11

Testing conv_backward_fast:
Naive: 4.253692s
Fast: 0.006516s
Speedup: 652.809294x
dx difference:  1.383704034070129e-11
dw difference:  2.497142522392147e-13
db difference:  0.0
```

```python
# Relative errors should be close to 0.0
from cs231n.fast_layers import max_pool_forward_fast, max_pool_backward_fast
np.random.seed(231)
x = np.random.randn(100, 3, 32, 32)
dout = np.random.randn(100, 3, 16, 16)
pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

t0 = time()
out_naive, cache_naive = max_pool_forward_naive(x, pool_param)
t1 = time()
out_fast, cache_fast = max_pool_forward_fast(x, pool_param)
t2 = time()

print('Testing pool_forward_fast:')
print('Naive: %fs' % (t1 - t0))
print('fast: %fs' % (t2 - t1))
print('speedup: %fx' % ((t1 - t0) / (t2 - t1)))
print('difference: ', rel_error(out_naive, out_fast))

t0 = time()
dx_naive = max_pool_backward_naive(dout, cache_naive)
t1 = time()
dx_fast = max_pool_backward_fast(dout, cache_fast)
t2 = time()

print('\nTesting pool_backward_fast:')
print('Naive: %fs' % (t1 - t0))
print('fast: %fs' % (t2 - t1))
print('speedup: %fx' % ((t1 - t0) / (t2 - t1)))
print('dx difference: ', rel_error(dx_naive, dx_fast))
```

```
Testing pool_forward_fast:
Naive: 0.012747s
fast: 0.003125s
speedup: 4.079118x
difference:  0.0

Testing pool_backward_fast:
Naive: 0.021447s
fast: 0.014788s
speedup: 1.450279x
dx difference:  0.0
```

## Three-layer ConvNet

在这里需要构建一个conv - relu - 2x2 max pool - affine - relu - affine - softmax这样的网络

- 随机数生成总结（https://blog.csdn.net/zenghaitao0128/article/details/78556535）
  - np.random.rand()函数

  > 通过本函数可以返回一个或一组服从**“0~1”均匀分布**的随机样本值。随机样本取值范围是[0,1)，不包括1。
  - np.random.randn()函数

  >通过本函数可以返回一个或一组服从**标准正态分布**的随机样本值。

  - numpy.random.randint(low, high=None, size=None, dtype=’l’)

  >返回随机整数，范围区间为[low,high），包含low，不包含high
  - np.random.normal(loc=loc, scale=weight_scale, size=size)

  >按照给定的size返回均值为loc，标准差为weight_scale的正态分布\

- 不太懂的一点，为什么如果是随机权重初始化，期望的交叉熵损失会是log(C)呢？

  After you build a new network, one of the first things you should do is sanity check the loss. When we use the softmax loss, we expect the loss for random weights (and no regularization) to be about `log(C)` for `C` classes. When we add regularization this should go up.

### 

```python
class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #                           
        ############################################################################
        C, H, W = input_dim
        F, HH, WW = num_filters, filter_size, filter_size
        #self.params['W1'] = np.random.normal(loc=loc, scale=weight_scale, size=W1_size)
        self.params['W1'] = weight_scale * np.random.randn(F, C, HH, WW)
        # 要把F个特征图全都tile 开来，所以神经元的个数就相当于F*高*宽
        self.params['W2'] = weight_scale * np.random.randn(F*H//2*W//2, hidden_dim)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b1'] = np.zeros(F)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['b3'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        pool_out, pool_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        affine_out, affine_cache = affine_relu_forward(pool_out, W2, b2)
        scores, cache = affine_forward(affine_out, W3, b3)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dscore = softmax_loss(scores, y)
        daffine, grads['W3'], grads['b3'] = affine_backward(dscore, cache)
        dpool, grads['W2'], grads['b2'] = affine_relu_backward(daffine, affine_cache)
        
        dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(dpool, pool_cache)

        loss += 0.5 * self.reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W3 ** 2))
        grads['W1'] += self.reg * W1
        grads['W2'] += self.reg * W2
        grads['W3'] += self.reg * W3
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


```

## Spatial Batch Normalization

这里是针对cnn的结构，提出新的bn的方式，使得维度适合cnn的输入就ok，所以主要的实现板块很简单，直接套用之前写完的部分，因为bn本身就是基于每一个batch的统计量，所以空间的bn也是需要将N那一维度的拿来作为第0维度输入，就可以套用之前的函数

```python
def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = x.shape
    x = x.transpose(0, 2, 3, 1).reshape(N * H * W, C)
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = dout.shape
    dout = dout.transpose(0, 2, 3, 1).reshape(N * H * W, C)
    dx, dgamma, dbeta = batchnorm_backward_alt(dout, cache)
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta
```

## Group Normalization

考虑完了空间结构，还需要考虑一下图像这个结构本身，边缘的像素对于整张图片的理解并没有太大帮助，意味着在同一层边缘的神经元往往不会被激活，所以一张图（即卷积中某一层）的差异很大，所以想到对于每一层，分成很多组去group normalization，削减数据之间的极大差异性。

![屏幕快照 2018-12-12 下午9.26.34](https://ws2.sinaimg.cn/large/006tNbRwly1fy4a6w6r74j30wu0cegr9.jpg)

这种高维度的操作还是觉得很吃力，下面全程参考别人的实现过程，整理一下思路上的点

- `x_group = np.reshape(x, (N, G, C//G, H, W)) `这里的分组手段没有想到，可以想象单维度的向量被分解成一个矩阵，然后在矩阵的某一维度上进行操作，从而做到了合理的分组
- 反向的过程和之前的没有任何区别，细微的区别就在于细节上
  - `N_GROUP = C//G*H*W`
    `dmean1 = np.sum(dx_groupnorm * -1.0 / np.sqrt(var + eps), axis=(2,3,4), keepdims=True)`
    `dmean2_var = dvar * -2.0 / N_GROUP * np.sum(x_group - mean, axis=(2,3,4), keepdims=True)`
  - 之所以需要设这个N_group，是因为在mean和var的操作中，这些部分都参与了，可以想象所有的feature map 需要都tile才能和一个全链接层的向量相比较，所以涉及到的维度都需要去除。

```python
def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    N,C,H,W = x.shape
    x_group = np.reshape(x, (N, G, C//G, H, W)) #按G将C分组
    mean = np.mean(x_group, axis=(2,3,4), keepdims=True) #均值
    var = np.var(x_group, axis=(2,3,4), keepdims=True) #方差
    x_groupnorm = (x_group-mean)/np.sqrt(var+eps) #归一化
    x_norm = np.reshape(x_groupnorm, (N,C,H,W)) #还原维度
    out = x_norm*gamma+beta #还原C
    cache = (G, x, x_norm, mean, var, beta, gamma, eps)


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    N,C,H,W = dout.shape
    G, x, x_norm, mean, var, beta, gamma, eps = cache
    # dbeta，dgamma
    dbeta = np.sum(dout, axis=(0,2,3), keepdims=True)
    dgamma = np.sum(dout*x_norm, axis=(0,2,3), keepdims=True)
    # 计算dx_group，(N, G, C // G, H, W)
    # dx_groupnorm
    dx_norm = dout * gamma
    dx_groupnorm = dx_norm.reshape((N, G, C // G, H, W))
    # dvar
    x_group = x.reshape((N, G, C // G, H, W))
    dvar = np.sum(dx_groupnorm * -1.0 / 2 * (x_group - mean) / (var + eps) ** (3.0 / 2), axis=(2,3,4), keepdims=True)
    # dmean
    N_GROUP = C//G*H*W
    dmean1 = np.sum(dx_groupnorm * -1.0 / np.sqrt(var + eps), axis=(2,3,4), keepdims=True)
    dmean2_var = dvar * -2.0 / N_GROUP * np.sum(x_group - mean, axis=(2,3,4), keepdims=True)
    dmean = dmean1 + dmean2_var
    # dx_group
    dx_group1 = dx_groupnorm * 1.0 / np.sqrt(var + eps)
    dx_group2_mean = dmean * 1.0 / N_GROUP
    dx_group3_var = dvar * 2.0 / N_GROUP * (x_group - mean)
    dx_group = dx_group1 + dx_group2_mean + dx_group3_var

    # 还原C得到dx
    dx = dx_group.reshape((N, C, H, W))
```

