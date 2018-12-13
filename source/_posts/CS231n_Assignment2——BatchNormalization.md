---
title: CS231n_Assignment2——BatchNormalization
date: 2018-12-07 21:23:25
tags: [CV, cs231n, assignment]
---

写完第一个ipynb后，开始写各种优化后的ipynb,这一篇是针对于会出现网络内部分布变化的内容进行设计了一个batch norm的层，旨在解决神经网络训练难的问题，以下是作业完成时参考的链接。

https://zhuanlan.zhihu.com/p/33173246

https://wwdguu.github.io/2018/05/01/normalization-in-deeplearning/

<!--more-->



## Batch normalization: forward

- 在训练阶段，采取的是对于每个batch的mean和var进行**移动平均**的做法，并且这个值会不断更新，并在测试阶段直接使用
- 但是对于这个平均值的定义还是有些不太理解，为什么要加上momentum这个参数，并且为什么这个参数会是为0.9呢？
- 还有一个细节在注释里，即什么时候更新这个移动平均值，我一开始的时候放置位置出错了，实际上是对于输入的数据理解上存在偏差

```python
def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        
        sample_mean = np.mean(x, axis = 0)
        sample_var = np.var(x, axis = 0)
        # 不要忘记eps,以后这些除法运算都会加入这个参数以使得分母不会因为主参数为0 而出现参数溢出的情况     
        x1 = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = x1 * gamma + beta
        
        cache = (x, gamma, beta, x1, sample_mean, sample_var, eps)
        # 我第一遍在做的时候是将更新操作放在了一开始，实际上我们在进行计算的时候是针对现在输入的这波batch 里面，所以我们计算上面的公式的时候输入的mean和var也应该是这个batch 里面的，所以才应该在之后更新
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        x1 = (x - running_mean) / np.sqrt(running_var + eps)
        out = x1 * gamma + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache
```

## Batch normalization: backward

- batchnorm_backward_alt是接下来的一个函数，是说要简化求导的

- 取tuple中的元素的方法：

  ```python
  >>> cache = (1,2)
  >>> cache
  (1, 2)
  >>> a,b =cache
  >>> a
  1
  >>> b
  2
  ```

- 一个问题是求导的维度问题；比如：out = x1 * gamma + beta ，维度情况是(N,D) * (D,) + (D,) = (N,D)，那么求导dgamma的时候，理论上 dgamma = dout * x1，但是等式右边的维度是(N,D)  * (N,D)=（N,D),而等式左边是(D,)，明显维度就不对等，所以我们操作的时候就是用 dgamma = np.sum(dout * x1, axis = 0) 来做，可是这样子为什么对呢？这个问题就扩展到对于遵循broadcast的公式，求导时怎么办呢？

- 如果上诉做法是对的，那我这里需要总结两种求导的公式

  - sample_mean = np.mean(x, axis = 0)，dx = dsample_mean * np.ones_like(x) / N
  - sample_var = np.var(x, axis = 0) ,dx = dsample_var * np.ones_like(x) / N * 2 * (x-sample_mean)
  - 以上都用到了np.ones_like(x) / N来扩充了维度，一开始我在计算时没有想到这一点，一直不知道怎么将(D,)的倒数扩展到(N,D)上面去

- 另一个问题就是求导时会忽略x这条路，以后自己在计算求导的时候，记得画一张图，以避免自己漏掉某一条分路

  ![WechatIMG1](https://ws3.sinaimg.cn/large/006tNbRwly1fxzjijj20ej30wq0poqe4.jpg)

```python
def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # 取tuple的元素
    x, gamma, beta, x1, sample_mean, sample_var, eps = cache
    N = x.shape[0]
    #sample_mean = np.mean(x, axis = 0) #(D,)
    #sample_var = np.var(x, axis = 0) #(D,)
             
    #x1 = (x - sample_mean) / np.sqrt(sample_var + eps) # [(N,D)-(,D)]/(D,) = (N,D)
    #out = x1 * gamma + beta # (N,D) * (D,) + (D,) = (N,D)
    
    dx1 = dout * gamma #(N, D) = (N,D) * (D,)
    dgamma = np.sum(dout * x1, axis = 0) # (D,) = sum( (N,D)  * (N,D) ,axis =0)???
    dbeta = np.sum(dout, axis = 0)# (D,)  =sum ((N,D) ,axis =0)????
    
    
    dsample_var = np.sum(-dx1 *(x - sample_mean) * 0.5 * ((sample_var + eps) ** -1.5), axis =0) #(D,)
    dx = dsample_var * np.ones_like(x) / N * 2 * (x-sample_mean)
    
    dsample_mean = np.sum(-dx1/ np.sqrt(sample_var + eps), axis =0)
    # 这一步我也是没想到，因为var计算是要用到mean的，所以画流程图时要注意呀！
    dsample_mean += dsample_var * np.sum(-2 * (x - sample_mean), axis = 0) / N
    dx += dsample_mean * np.ones_like(x) / N
    
    dx += ((sample_var+ eps) ** -0.5) * dx1
    
    #dx = dsample_mean * np.ones_like(x) / N
    #dx += dsample_var * np.ones_like(x) / N * 2 * (x-sample_mean)
    
    # dsample_var = dx_3_b =dx_4_b
    # (dx +=)  =  dx_6_b
    # dsample_mean = dx_2_a
    #dsample_mean = np.sum(-dx1/ np.sqrt(sample_var + eps), axis =0)# (D,) = sum( (N,D)/(D,),axis = 0)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta
```

## Fully Connected Nets with Batch Normalization

- 字典推导式：

  ```python
  >>> a = [{'mode': 'train'} for i in range(5)]
  >>> a
  [{'mode': 'train'}, {'mode': 'train'}, {'mode': 'train'}, {'mode': 'train'}, {'mode': 'train'}]
  ```

- 出现的两个问题

  - 因为多层的关系，存储参数的变量要变成列表或者字典，这就需要初始化，一开始我初始化为np.zeros(self.num_layets),出现的问题就是，这种初始化是ndarry，也就是是矩阵，如果你之后赋值进去的东西是不同维度的，就会存在问题，所以后来才用了 [0] * self.num_layers，列表用来存储不同维度的元素是没有问题的

  - 注意前向推导和反向推导的时候，顺序是不一样的，所以正向

    ```python
    for i in range(self.num_layers-1):
    ```

    反向的要

    ```python
    for n in range(self.num_layers-1):
    	i = self.num_layers-2- n
    ```

```python
class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
                        
        for i in range(self.num_layers - 1):
            W_size = (hidden_dims[i-1], hidden_dims[i])
            b_size = (hidden_dims[i],)    
            if i == 0:
                W_size = (input_dim,hidden_dims[0])                
            self.params['W' + str(i+1)] = np.random.normal(loc=0.0, scale=weight_scale, size=W_size)   
            self.params['b' + str(i+1)] = np.zeros(b_size)              
            if self.normalization:
                self.params['gamma' + str(i+1)] = np.ones((hidden_dims[i],))
                self.params['beta' + str(i+1)] = np.zeros((hidden_dims[i],))
                
         
        self.params['W' + str(self.num_layers)] = np.random.normal(loc=0.0, scale=weight_scale, size= (hidden_dims[-1], num_classes))   
        self.params['b' + str(self.num_layers)] = np.zeros((num_classes,))  
                
            
        #print(self.params)
         
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################

        #hidden_out, caches = np.zeros(self.num_layers), np.zeros(self.num_layers)
        hidden_out, caches = [0] * self.num_layers, [0] * self.num_layers
        #print(type(hidden_out))

        
        for i in range(self.num_layers-1):
            w, b = self.params['W'+str(i+1)], self.params['b'+str(i+1)]
            
            if self.normalization == 'batchnorm': 
                gamma, beta = self.params['gamma'+str(i+1)], self.params['beta'+str(i+1)]
                if self.use_dropout:
                    if i == 0:
                        hidden_out[i], caches[i] = affine_bn_relu_drop_forward(X, w, b, gamma, beta, self.bn_params[i], dropout_param[i])   
                    else:
                        hidden_out[i], caches[i] = affine_bn_relu_drop_forward(hidden_out[i-1], w, b, gamma, beta, self.bn_params[i], dropout_param[i])
                    
                else:
                    if i == 0:
                        hidden_out[i], caches[i] = affine_bn_relu_forward(X, w, b, gamma, beta, self.bn_params[i])   
                    else:
                        hidden_out[i], caches[i] = affine_bn_relu_forward(hidden_out[i-1], w, b, gamma, beta, self.bn_params[i])
                
                
                    
            elif self.normalization == 'layernorm':
                gamma, beta = self.params['gamma'+str(i+1)], self.params['beta'+str(i+1)]
                if i == 0:
                    hidden_out[i], caches[i] = affine_ln_relu_forward(X, w, b, gamma, beta, self.ln_params[i])   
                else:
                    hidden_out[i], caches[i] = affine_ln_relu_forward(hidden_out[i-1], w, b, gamma, beta, self.ln_params[i]) 
                                    
            else:
                if i == 0:
                    hidden_out[i], caches[i] = affine_relu_forward(X, w, b) 
                else:
                    hidden_out[i], caches[i] = affine_relu_forward(hidden_out[i-1], w, b) 
                
        w, b = self.params['W'+str(self.num_layers)], self.params['b'+str(self.num_layers)]
        hidden_out[self.num_layers - 1], caches[self.num_layers - 1] = affine_forward(hidden_out[self.num_layers - 2], w, b)

        scores = hidden_out[self.num_layers-1]
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)
             
        dhidden_out = [1] * self.num_layers
        dhidden_out[self.num_layers-1] = dscores
        
        dx, dw, db = affine_backward(dscores, caches[self.num_layers-1])
        dhidden_out[self.num_layers-2], grads['W' + str(self.num_layers)], grads['b'+str(self.num_layers)] = dx, dw, db      
        loss += 0.5 * self.reg * np.sum(self.params['W' + str(self.num_layers)] ** 2)
        grads['W' + str(self.num_layers)] += self.reg * self.params['W' + str(self.num_layers)]
   
        
        for n in range(self.num_layers-1):
            i = self.num_layers-2- n
            if self.normalization == 'batchnorm':
                if self.use_dropout:
                    dx, dw, db, dgamma, dbeta = affine_bn_relu_drop_backward(dhidden_out[i], caches[i])
                    grads['gamma'+str(i+1)], grads['beta' + str(i+1)] = dgamma, dbeta
                    if i ==0:
                        dX, grads['W' + str(i+1)], grads['b' + str(i+1)] = dx, dw, db
                    else:  
                        dhidden_out[i-1], grads['W' + str(i+1)], grads['b' + str(i+1)] = dx, dw, db
                else:
                    dx, dw, db, dgamma, dbeta = affine_bn_relu_backward(dhidden_out[i], caches[i])
                    grads['gamma'+str(i+1)], grads['beta' + str(i+1)] = dgamma, dbeta
                    if i ==0:
                        dX, grads['W' + str(i+1)], grads['b' + str(i+1)] = dx, dw, db
                    else:  
                        dhidden_out[i-1], grads['W' + str(i+1)], grads['b' + str(i+1)] = dx, dw, db
                
            else:
                dx, dw, db = affine_relu_backward(dhidden_out[i], caches[i])
                if i ==0:
                    dX, grads['W' + str(i+1)], grads['b' + str(i+1)] = dx, dw, db
                else:
                    dhidden_out[i-1], grads['W' + str(i+1)], grads['b' + str(i+1)] = dx, dw, db

            loss += 0.5 * self.reg * np.sum(self.params['W' + str(i+1)] ** 2)
            grads['W' + str(i+1)] += self.reg * self.params['W' + str(i+1)]
        

        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
```

### Batchnorm for deep networks

总结亮点：

1、bn对于解决权重初始化有着重要的作用，即缓解权重初始化某些值对网络起不到作用的功能（表述不是很好，意思到位了，可是还是不是很明白为什么），参考链接**https://www.leiphone.com/news/201703/3qMp45aQtbxTdzmK.html**

![屏幕快照 2018-12-10 下午5.37.22](https://ws2.sinaimg.cn/large/006tNbRwly1fy1sc91xfjj30w60u0dng.jpg)

2、bn的batch越小，单个batch的均值和方差就越不稳定，使得效果会很差，batch大的话内存又不一定够用，所以很依赖batch_size的选取

![屏幕快照 2018-12-10 下午5.37.38](https://ws3.sinaimg.cn/large/006tNbRwly1fy1sccg7g1j310k0o60yu.jpg)

### Layer Normalization

为了解决batchnorm的问题，有人提出了layernorm,本质上是做高斯变换的维度变了，变成了针对于channel，而不是对number，这样子就可以有效规避batch所带来的影响。

那么在编码的时候，只需要注意跟换下维度，就可以复制bn的代码来做，但是效果一般没有bn那么好，但是可以用在rnn上。

![屏幕快照 2018-12-10 下午5.48.41](https://ws3.sinaimg.cn/large/006tNbRwly1fy1snz553tj310l0u0dn1.jpg)

```python

def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    x_T = x.T#(D,N)
    
    sample_mean = np.mean(x_T, axis=0)# (N,)
    sample_var = np.var(x_T, axis=0)# (N,)
    x_norm_T = (x_T - sample_mean) / np.sqrt(sample_var + eps) #(D,N)
    x_norm = x_norm_T.T#(N,D)
    out = x_norm * gamma + beta#(N,D)
    
    cache = (x, x_norm, gamma, beta, sample_mean, sample_var, eps)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    x, x_norm, gamma, beta, sample_mean, sample_var, eps = cache
    D = x_T.shape[0]
    x_T = x.T#(D,N)
    
    
    dbeta = np.sum(dout, axis=0)# [(N,D),axis =0] = (D,)
    dgamma = np.sum(x_norm * dout, axis=0)# [(N,D)*(N,D)=(N,D),axis =0] = (D,)
    dx_norm = dout * gamma# (N,D) = (N,D)*(D,)
    dx_norm_T = dx_norm.T#(D,N)
    
    dsample_var = np.sum(-dx_norm_T *(x_T - sample_mean) * 0.5 * ((sample_var + eps) ** -1.5), axis =0) #(D,)
    dx_T = dsample_var * np.ones_like(x_T) / D * 2 * (x_T-sample_mean)
    
    dsample_mean = np.sum(-dx_norm_T/ np.sqrt(sample_var + eps), axis =0)
    dsample_mean += dsample_var * np.sum(-2 * (x_T - sample_mean), axis = 0) / D
    dx_T += dsample_mean * np.ones_like(x_T) / D    
    dx_T += ((sample_var+ eps) ** -0.5) * dx_norm_T
    
    dx = dx_T.T
    
   

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
```

