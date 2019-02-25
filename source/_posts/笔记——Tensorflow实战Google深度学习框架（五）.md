---
title: 笔记——Tensorflow实战Google深度学习框架（五）
date: 2018-12-22 15:13:07
tags: [CV, 笔记, tensorflow]
---

# 第 5 章 MNIST 数字识别问题

<!--more-->

## 5.1 MNIST 数据处理

- 含三个数据集：train，validation，test，其中train有55000张，validation有5000张，test有10000张
- 每张图片维度（28，28），在数据集中被展成长度为784的一维数组

```python
from tensorflow.examples.tutorials.mnist import input_data
# 一个dataset类
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# 查看训练数据的大小
print(mnist.train.images.shape)  # (55000, 784)
print(mnist.train.labels.shape)  # (55000, 10)
 
# 查看验证数据的大小
print(mnist.validation.images.shape)  # (5000, 784)
print(mnist.validation.labels.shape)  # (5000, 10)
 
# 查看测试数据的大小
print(mnist.test.images.shape)  # (10000, 784)
```

- 为了使用随机梯度下降，则需要批量读取数据`mnist.train.next_batch(batch_size)`

```python
batch_size1 = 10
xs, ys = mnist.train.next_batch(batch_size1)
xs2, ys2 = mnist.train.next_batch(batch_size1)

print(xs.shape)
print(xs2.shape)
print(ys.shape)
print(xs is xs2)
# (10, 784)
# (10, 784)
# (10, 10)
# False
```

- 从上面的测试可以看出每次的batch是不一样的结果，所以在真正使用中，应该是

```python
for i in range(total_batch_num):
    example_batch，label_batch = mnist.next_batch(batch_size)
```

## 5.2 神经网络模型训练及不同模型结果对比

### 5.2.1 TensorFlow 训练神经网络

这里是一个完整的mnist数据集的内容，下面摘录一些需要关注的tf函数

- `cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))`

  - 当中tf.argmax(y_,1),第一参数是对象y__(n,d)，第二个参数是求最大值的是哪个维度，求完后结果是(n,1)，那么这时候相当于每个label都是个单一的值，不是独热向量了，所以可以用sparse的交叉熵损失
- `global_step = tf.Variable(0, trainable=False)`，这里的global_step用在了两个地方，一个是滑动平均值处，一个是衰减学习率那里，虽然没有显示透出这个参数的更新过程，但是要记得，在`    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)`这里，就点名了这个global_step 会自动更新

- 组合两种操作的方式：

  - 一是用控制依赖，并用`tf.no_op(name='train')`作为一个什么操作都没有的节点；

  ```python
  # tf.no_op()表示执行完 train_step, variable_averages_op 操作之后什么都不做
  with tf.control_dependencies([train_step, variables_averages_op]：
  	train_op = tf.no_op(name='train')
  # ..省略中间步骤....
  sess.run(train_op,feed_dict={x:xs,y_:ys})
  ```

  - 二是用`tf.group()`

  ```python
  train_op = tf.group(train_step, variables_averages_op)
  # ..省略中间步骤....
  sess.run(train_op,feed_dict={x:xs,y_:ys})
  ```

- 计算正确率的方式之一

```python
    # 计算正确率，返回一个元素为布尔变量的矩阵
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    # tf.cast()函数的作用是执行 tensorflow 中张量数据类型转换
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```

```python
INPUT_NODE = 784     # 输入节点
OUTPUT_NODE = 10     # 输出节点
LAYER1_NODE = 500    # 隐藏层数       
                              
BATCH_SIZE = 100     # 每次batch打包的样本个数        

# 模型相关的参数
LEARNING_RATE_BASE = 0.8      
LEARNING_RATE_DECAY = 0.99    
REGULARAZTION_RATE = 0.0001   
TRAINING_STEPS = 5000        
MOVING_AVERAGE_DECAY = 0.99

def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 不使用滑动平均类
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2

    else:
        # 使用滑动平均类
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)
        
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    # 生成隐藏层的参数。
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # 生成输出层的参数。
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算不含滑动平均类的前向传播结果
    y = inference(x, None, weights1, biases1, weights2, biases2)
    
    # 定义训练轮数及相关的滑动平均类 
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)
    
    # 计算交叉熵及其平均值
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
    # 损失函数的计算
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    regularaztion = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularaztion
    
    # 设置指数衰减的学习率。
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)
    
    # 优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
    # 反向传播更新参数和更新每一个参数的滑动平均值
    # 这里也可以用tf.group()，组合两个op
    with tf.control_dependencies([train_step, variables_averages_op]):
        # tf.no_op()表示执行完 train_step, variable_averages_op 操作之后什么都不做
        train_op = tf.no_op(name='train')

    # 计算正确率
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    # tf.cast()函数的作用是执行 tensorflow 中张量数据类型转换
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # 初始化会话，并开始训练过程。
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels} 
        
        # 循环的训练神经网络。
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy using average model is %g " % (i, validate_acc))
            
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})

        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print(("After %d training step(s), test accuracy using average model is %g" %(TRAINING_STEPS, test_acc)))

def main(argv=None);
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)	
    train(mnist)
if __name__ == '__main__':
    tf.app.run()
```

### 5.2.2 使用验证数据 集判断模型效果

- 使用验证集的一个前提是验证集和测试集的分布最好是一致的
- 由于mnist数据较为简单，所以当我们利用这几种技巧去优化的时候，往往因为梯度在很早就趋于稳定而显现不出来优势，实际上**学习率的衰减、滑动平均值、正则化都是很有效的方式**（前两者主要在限制参数更新的速度，因为mnist更新过快，所以导致效果不明显）![屏幕快照 2018-12-22 下午6.53.20](https://ws3.sinaimg.cn/large/006tNbRwly1fyfpyscnqqj30s60cw45r.jpg)

## 5.3 变量管理

- 更好的变量管理方式，取代了引入很多变量的写法

- 例子：从`def inference (input_tensor, avg_class, weightsl, biases1, weights2, biases2)`

变成`def inference(input_tensor, reuse=False} `

- `tf.variable_scope`和`tf.get_variable`结合使用

  - `tf.get_Variable`和`tf.Variable`	的区别在于前者的变量名称是必须指定的

    ```python
    # 这两者是等价的
    v = tf.get_Variable (”v”, shape=[l], initializer=tf.constant_initializer(l.0)) 
    v = tf.Variable(tf.constant(l.0 , shape=[l]), name=” v” )
    ```

  - reuse参数的指定，表明参数可以复用，有了这个机制，就可以在第一次生成网络时设置为False，后面不断训练的时候使用True

    ```python
    with tf.variable_scope("foo"):
        v = tf.get_variable("v", [1], initializer=tf.constant_initializer(1.0))
        
    with tf.variable_scope("foo", reuse=True):
        v1 = tf.get_variable("v", [1])
    print (v == v1)
    # True
    ```

  - `tf.variable_scope`是可以不断嵌套的，与此同时，`tf.variable_scope`生成的上下文管理器会产生一个命名空间，在这个空间里生成的变量名称都会带上这个命名空间名作为前缀

    ```python
    v1 = tf.get_variable("v", [1])
    print (v1.name)
    # v:0
    
    with tf.variable_scope("foo",reuse=True):
        v2 = tf.get_variable("v", [1])
    print (v2.name)
    #foo/v:0
        
    with tf.variable_scope("foo"):
        with tf.variable_scope("bar"):
            v3 = tf.get_variable("v", [1])
            print (v3.name)
    #foo/bar/v:0
    
    v4 = tf.get_variable("v1", [1])
    print (v4.name)
    #v1:0
    ```

  - 可以通过变量的名称来获取变量

    ```python
    with tf.variable_scope("",reuse=True):
        v5 = tf.get_variable("foo/bar/v", [1])
        print (v5 == v3)
        # True
        v6 = tf.get_variable("v1", [1])     
        print (v6 == v4)
        # True
    ```

- 下面是更改后的完整案例

  - 注意第一次设置reuse为False，初次构建整个网络模型，以后使用设置为True
  - ![屏幕快照 2018-12-23 下午3.11.11](https://ws4.sinaimg.cn/large/006tNbRwly1fygpb6kkofj30ry0pa1i1.jpg)

## 5.4 TensorFlow 模型持久化

`tf.train.Saver`实现保存和加载模型，模型分为三个

- 第一个文件为 model.ckpt.meta：保存了图的结构
- 第二个文件为 model.ckpt：保存了变量的取值
- 第三个文件为 checkpoint 文件：保存了文件中一个目录下所有模型文件的列表（？？？有点不太懂）

### 5.4.1 持久化代码实现

```python
# 保存计算两个变量和的模型
v1 = tf.Variable(tf.random_normal([1], stddev=1, seed=1))
v2 = tf.Variable(tf.random_normal([1], stddev=1, seed=1))
result = v1 + v2

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    saver.save(sess, "Saved_model/model.ckpt")
    

```

```python
# 加载保存了两个变量和的模型    
with tf.Session() as sess:
	saver.restore(sess, "Saved_model/model.ckpt")
	print(sess.run(result))
```

- 可以直接加载持久化的图，相当于把之前静态图的设计部分用这个代替了

```python
# 直接加载持久化的图
saver = tf.train.import_meta_graph("Saved_model/model.ckpt.meta")
with tf.Session() as sess :
	saver.restore(sess, ”/path/to/model/model.ckpt”) #通过张茸的名称来获取张量。
	print(sess.run(tf.get_default_graph().get_tensor_by name(”add:0”)))#输出[ 3.]
```

- 也可以加载部分模型,利用列表选择要加载的内容`saver = tf. train. Saver([v1]`

- 还可以重命名

```python
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name = "other-v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name = "other-v2")
# 将原来名称为v1的变量加载到名称为"other-v1"的变量中去，这样的话就把名称为v1的变成了名称为other-v1了
saver = tf.train.Saver({"v1": v1, "v2": v2})
```

- 可以把重命名的技巧放在滑动平均值的计算上

```python
# 使用滑动平均
v = tf.Variable(0, dtype=tf.float32, name="v")
for variables in tf.global_variables(): print(variables.name)
# v:0
ema = tf.train.ExponentialMovingAverage(0.99)
maintain_averages_op = ema.apply(tf.global_variables())
for variables in tf.global_variables(): print(variables.name)
# v:0
# v/ExponentialMovingAverage:0

# 保存滑动平均
saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    sess.run(tf.assign(v, 10))
    sess.run(maintain_averages_op)
    # 保存的时候会将v:0  v/ExponentialMovingAverage:0这两个变量都存下来。
    saver.save(sess, "Saved_model/model2.ckpt")
    print(sess.run([v, ema.average(v)]))
    # [10.0, 0.099999905]
    
# 加载滑动平均模型
# v = tf.Variable(0, dtype=tf.float32, name="v")

# 通过变量重命名将原来变量v的滑动平均值直接赋值给v。
saver = tf.train.Saver({"v/ExponentialMovingAverage": v})
with tf.Session() as sess:
    saver.restore(sess, "Saved_model/model2.ckpt")
    print sess.run(v)
```

- 为了方便使用滑动平均变量，`tf.train.ExponentialMovingAverage `类提供了
  `variables_to_restore` 函数来生成` tf.train.Saver`类所需要的变量重命名字典。

```python
import tensorflow as tf
v = tf.Variable(0, dtype=tf.float32, name="v")
ema = tf.train.ExponentialMovingAverage(0.99)
# print(ema.variables_to_restore())
# {u'v/ExponentialMovingAverage': <tf.Variable 'v:0' shape=() dtype=float32_ref>}


saver = tf.train.Saver(ema.variables_to_restore())
with tf.Session() as sess:
    saver.restore(sess, "Saved_model/model2.ckpt")
    print(sess.run(v)) 
    #INFO:tensorflow:Restoring parameters from Saved_model/model2.ckpt
    #0.0999999
```

- 当测试的时候，有时候不需要一些辅助的节点，往往只需要直接直接求出前向传播的结果就好,所以tf提供了一个`convert_variables_to_constants `函数，这样的话就可以把计算图的变量作为常数保存起来，不用分开保存了

  - 在读取模型文件获取变量的值的时候，我们需要指定的是**张量的名称**而不是**节点的名称**
  - 通过convert_variables_to_constants函数来指定保存的**节点名称**而不是**张量的名称**

  ```python
  # pb文件的保存
  import tensorflow as tf
  # graph_util:在 python 中操作张量图的帮助器
  from tensorflow.python.framework import graph_util
  
  v1 = tf.Variable(tf.constant(1.0, shape=[1]), name = "v1")
  v2 = tf.Variable(tf.constant(2.0, shape=[1]), name = "v2")
  result = v1 + v2
  
  init_op = tf.global_variables_initializer()
  with tf.Session() as sess:
      sess.run(init_op)
      # 返回一个图的序列化的GraphDef表示，序列化的GraphDef可以导入至另一个图中(使用 import_graph_def())
      graph_def = tf.get_default_graph().as_graph_def()
      # convert_variables_to_constants函数来指定保存的节点名称而不是张量的名称，“add:0”是张量的名称而"add"表示的是节点的名称。
      output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])
      with tf.gfile.GFile("Saved_model/combined_model.pb", "wb") as f:
             f.write(output_graph_def.SerializeToString())
  # INFO:tensorflow:Froze 2 variables.
  # Converted 2 variables to const ops.
  ```

  ```python
  # 加载pb文件
  from tensorflow.python.platform import gfile
  with tf.Session() as sess:
      model_filename = "Saved_model/combined_model.pb"
     
      with gfile.FastGFile(model_filename, 'rb') as f:
          graph_def = tf.GraphDef()
          graph_def.ParseFromString(f.read())
  # 在读取模型文件获取变量的值的时候，我们需要指定的是张量的名称而不是节点的名称
      result = tf.import_graph_def(graph_def, return_elements=["add:0"])
      print sess.run(result)
  ```

### 5.4.2 持久化原理及数据格式

接下来对于存储的3个文件进行详细地分析

- .meat文件：通过元图（MetaGraph）来`计算图中节点的信息`和`计算图中所需要的元数据`，MetaGraph本身是用protocol Buffer 来定义的

```python
message MetaGraphDef { 
	MetainfoDef meta_info_def = 1;
	
	GraphDef graph_def = 2 ;
	SaverDef saver_def = 3;
	map<string , CollectionDef> collection_def = 4 ; 
	map<string, SignatureDef> signature_def = 5; 
	repeated AssetFileDef asset_file_def = 6;
	}
```

- 一般用了`saver = tf.train_Saver()`后，保存的数据都是二进制的，所以在这里用`export_meta_graph`函数将数据导成json格式

  ```python
  import tensorflow as tf
  #定义变量相加的计算。
  vl = tf.Variable(tf.constant(1.0, shape=[l]), name=”vl”) 
  v2 = tf.V ariable (tf.constant(2.0 , shape=[l]) , name=”v2”) 
  resultl = vl + v2
  
  saver = tf.train.Saver()
  #通过 export meta graph 函数导出 TensorFlow 计算固的元圈，并保存为 json 格式。 
  saver.export_meta_graph ( ”/path/to/model.ckpt.meda.json ”， as_text=True)
  ```

  - **meta_info_def属性**

    记录了计算图中的元数据（计算图的版本号+用户指定的标签，如果在saver时没有特别指定，那么这些属性都默认为空）和所有运算方法的信息，是用`MetaInfoDef`来定义

    ```python
    #
    message MetainfoDef {
        # 计算图的版本号
        string meta_graph_version = l; 
        # 记录计算图上的所有运算的信息
        # 所以 一个运算即使被多次使用，也只会保存一个运算，比如Variable赋值这里有两次，但是只记录一个variable的信息
        OpList stripped_op_list = 2; 
        google.protobuf.Any_any_info = 3; 
        # 用户指定的标签
        repeated string tags = 4;
        # 下面两个都是计算图的tensorflow的版本信息
        string tensorflow_version = 5; 
    	string tensorflow_git_version = 6;
    }
    ```

    - OpList类型：`stripped_op_list`里面的运算就是这种类型的，是用`OpDef`来定义

      ```python
      message OpDef { 
          # 前四个属性是核心的
          # 运算的类型，唯一标识符
          string name = l ;
          # 运算的输入，因为可能有多个，所以会重复多个，是个列表（repeated）
      	repeated ArgDef input_arg = 2; 
          # 运算的输出，因为可能有多个，所以会重复多个，是个列表（repeated）
          repeated ArgDef output_arg = 3; 
          # 其他的参数信息
          repeated AttrDef attr = 4;
          
      	OpDeprecation deprecation = 8; 
          string summary = 5;
      	string description = 6;
      	bool is commutative = 18;
      	bool is aggregate = 16
      	bool is stateful = 17;
      	bool allows_uinitialized_input = 19;
      }
      ```

      给一个例子：

      ```python
      op {
          name : ”Add”
          input_arg { 
              name:”x”
          	type_attr:”T”
          }
          input_arg {
          	name:”Y” 
              type_attr:”T”
          }
          output_arg {
      		name: ”z” 
          	type_attr: ”T”
          }
          attr{
              name: "T"
      		type : ”type” 
              allowed values {
                  list{
                      type: DT_HALF
                      type: DT_FLOAT
                      ...
                  }
              }
          }
      }
      ```

  - graph_def属性

    主要记录了Tensorflow计算图上的节点信息，因为在 meta_info_def属性中已经定义了每种运算的具体信息，所以这里只需要记录运算节点之间的链接关系就好，其通过Protocol Buffer定义的`GraphDef`来表示，里面也包含了一个`NodeDef`类型的列表

    ```python
    message GraphDef {
        # 节点的信息
    	repeated NodeDef node = 1; 
        # 存储的是tensor flow的版本号
        VersionDef versions = 4;
        # 还有一些已经不用的或者还在试验中的属性
    }
    
    message NodeDef {
        # 唯一标识符，节点的名称属性
        string name = 1;
        # 运算方法的名称，通过这个去找meta_info_def里面的信息
        string op = 2;
        # 是一个字符串列表，取值格式为node:src_output,当后面的输出是这个节点的第一个输出的话可以省略，比如，node:0表示名称为 node 的节点的第一个输出，它也可以被记为 node。
        repeated string input =3;
        # 指定处理这个运算的设备，当为空时，自动选取一个最合适的设备来做
        string device =4;
        # 指定了与当前运算相关的运算信息，map指的是一个键值对
        map<string,AttrValue> attr = 5;
    }
    ```

    给出一个例子

    ```python
    graph def {
        # 变量定义运算
        node {
    		name:”vl” 
            op:”VariableV2” 
            # 定义了维度
            attr {
    			key:”_output_shapes” 
                value { 
                    list{ shape { dim { size: 1 } } }
                }
            }
            # 定义了类型
            attr {
    			key:”dtype” 
                value {
    				type: DT FLOAT
                }
            }
    	...
        # 因为节点后面用到的是第一个输出，就省略了后面的内容
    	node{
            name:"add"
            op:"Add"
            input:"v1/read"
            input:"v2/read" 
    	}
    	# 数据持久化后自动完成的运算
        node{
            name:"save/control_dependency"
            op:"Identity"
            ...
        }
        versions{
            producer:24
        }
    }
    ```

  - saver_def属性

    记录了数据持久化需要用到的一些参数，包括保存到文件的文件名、保存操作和 加载操作的名称以 及保存频率 、 清理历史记录等。用`SaverDef`来定义

    ```python
    message SaverDef {
        # 保存文件名的张量名称
    	string filename_tensor_name = l;
        # 持久化模型的运算所对应的节点名称
        string save_tensor_name = 2; 
        # 加载模型的运算
        string restore_op_name = 3;
        # 设定最大保存的量，即超过设定的数量后第一次保存的模型就会删除
        int32 max_to_keep = 4;
    	bool sharded = 5;
        # 每n个小时在max_to_keep的基础上多保存一个模型
    	float keep_checkpoint_every_n_hours = 6;
        
    	enum CheckpointFormatV ersion {
            LEGACY = 0 ;
        	Vl= l;
        	V2 = 2;
        }
        CheckpointFormatVersion version = 7;
    }
    ```

    例子

    ```python
    saver def {
        filename_tensor_name: ”save/Const:0” 
        save_tensor_name:”save/control_dependency:0” 
        restore_op_name:”save/restore_all”
        max_to_keep: 5
        keep_checkpoint_every_n_hours:10000.0 
        version :V2
    }
    ```

  - collection def属性，是对于集合的维护，是一个从集合名称到集合内容的映射

    ```python
    message CollectionDef { 
        message NodeList {
    		repeated string value = 1;
        }
    	message BytesList { 
            repeated bytes value = 1;
        }
    	message Int64List {
    		repeated int64 value= 1 [packed= true];
        }
    	message FloatList {
    		repeated float value= 1 [packed= true];
        }
    	message AnyList {
    		repeated google.protobuf.Any value = 1;
        }
    	oneof kind {
    		NodeList node_list = 1;
            BytesList bytes_list = 2; 
            Int64List int64_list = 3; 
            FloatList float_list = 4; 
            AnyList any_list = 5;
        }
    }
    ```

    例子

    ![屏幕快照 2018-12-25 下午5.55.41](https://ws2.sinaimg.cn/large/006tNbRwly1fyj55miyxmj30se06owkn.jpg)

    ![屏幕快照 2018-12-25 下午5.55.51](https://ws1.sinaimg.cn/large/006tNbRwly1fyj55nzzizj30sa07gdmf.jpg)

- 第二类是存储变量的取值：`model.ckpt.index`和`model.ckpt.data-XXXX-of-XXXX`文件,其中后者文件是以SSTable格式存储的，可以简单地看作一个（key,value）列表，可以通过来`tf.train.NewCheckpointReader`查看

  ```python
  import tensorflow as tf
  # tf.train.NewCheckpointReader 可以读取 checkpoint 文件中保存的所有变量。
  # 注在后面的 .data 和 .index 可以省去。
  reader = tf.train.NewCheckpointReader (’/path/to/model/model.ckpt')
  #获取所有变盘列哀。这个是 一个从变量名到变量维度的字典。 
  global_variables = reader.get_variable_to_shape_map() 
  for variable_name in global_variables:
  # vaiiable name 为变吐名称， global var工ables[variable name]为变莹的维度。 
  	print(variable_name, global_variables[variable_name])
  #取名称为 vl 的变盘的取值。
  print (”Value for variable v1 is ”, reader.get_tensor (”vl”))
  ```

- 最后一类是checkpoint文件

  - 是`tf.train.Saver`类自动生成且自动维护的，维护了由 一个` tf.train.Saver` 类持久化的所有 TensorFlow 模型文件的文件名。

  - 当某个保存的 TensorFlow 模型文件被删除时 ， 这个模型所对应的文件 名也会从 checkpoint 文件中删除 。 

  - checkpoint 中内 容的格式为 CheckpointState Protocol Buffer，下面给出了 CheckpointState类型的定义。 

    ```python
    message CheckpointState {
        # 保存了最新的tensorflow文件名
    	string model_checkpoint_path = l ;
        # 列出了当前还没有被删除的所有 TensorFlow 模型文件的文件名
    	repeated string all_model_checkpoint_paths = 2;
    }
    ```

    如：

    ```python
    model checkpoint path:”/path/to/model/model.ckpt” all_model_checkpoint_paths : ”/ path/to/model/model.ckpt”
    ```

## 5.5 TensorFlow 最佳实践样例程序

三个存在的问题：

- 需要传入的参数太多，代码冗杂：通过get_variable来解决
- 跑完的模型没有持久化，以后没法复用
- 跑程序的时候容易服务器出问题，所以需要采取隔一段时间存储以下结果

重构：将前向传播部分 (inference) 、训练部分 (train) 、测试部分(eval)分开

```python
import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', regularizer(weights))
    return weights


def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):

        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2
```

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import os

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="MNIST_model/"
MODEL_NAME="mnist_model"


def train(mnist):

    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)


    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
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


    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets("../../../datasets/MNIST_data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()

```

```python
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import mnist_train

# 加载的时间间隔。
EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        y = mnist_inference.inference(x, None)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)
def main(argv=None):
    mnist = input_data.read_data_sets("../../../datasets/MNIST_data", one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    main()
```

