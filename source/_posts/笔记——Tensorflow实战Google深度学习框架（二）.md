---
title: 笔记——Tensorflow实战Google深度学习框架（二）
date: 2018-12-17 13:59:51
tags: [CV, 笔记, tensorflow]
---

Tensorflow环境搭建

<!--more-->

## 2.1 Tensorflow 主要依赖包

主要是Protocol Buffer 和Bazel

### 2.1.1 Protocol Buffer

- 定义
  - 是谷歌开发的**处理结构化数据**的工具
  - **处理结构化数据**的步骤：先将**结构化的数据**序列化，并从序列化的数据流中还原出原来的结构化数据
- 和XML，JSON区别
  - 序列化后的数据不是可读的字符串，是二进制流
  - XML，JSON可以从序列化后的数据直接还原，但是Protocol Buffer需要预先定义数据的格式，还原时需要用到，因此序列化后的数据相比数据量小3-10倍，解析时间快20-100倍。
- .proto：定义数据格式的文件
  - massage代表了一个类，包含字段限制,字段类型,字段名和编号
    - 字段类型：布尔，整数，也可以是另一个message
    - 字段限制：必须/可选/可重复（意味着取值可以是一个列表）
    - 字段名
    - 编号：不表示字段的值，只是表示第几个属性

```protobuf
message Person{
optional string name = 1;
required int32 id = 2;
repeated string email = 3;
}
```

- 一旦定义好数据格式后，可以运行protocol buffer编译器，它会基于.proto文件为应用程序的语言（C++等）生成相应的类

  你的代码可以这样写：

  ```c++
  Person person;
  person.set_name("John Doe");
  person.set_id(1234);
  person.set_email("jdoe@example.com");
  fstream output("myfile", ios::out | ios::binary);
  person.SerializeToOstream(&output);
  ```

  然后你可以这样来读回你的消息：

  ```c++
  fstream input("myfile", ios::in | ios::binary);
  Person person;
  person.ParseFromIstream(&input);
  cout << "Name: " << person.name() << endl;
  cout << "E-mail: " << person.email() << endl;
  ```

### 2.1.2 Bazel

- 定义：谷歌开源的编译软件

- 项目空间：其对应的文件夹是这个项目的根目录；包含源代码和输出编译结果的软连接WORKSPACE；可以包含一个或者多个应用

  - 必须有WORKSPACE文件：定义了对外部资源的依赖关系
  - BUILD文件：通过这个文件找到需要编译的目标；使用类似于python的语法指定每一个编译目标的输入、输出以及编译方式
  - 源代码

- 编译python文件：

  - py_binary：将python程序编译为可执行文件
  - py_library：将python程序编译为库函数以便py_binary和py_test调用
  - py_test：     编译python测试文件

- 编译实例：

  - 项目空间的文件如下：![屏幕快照 2018-12-17 下午8.22.54](https://ws3.sinaimg.cn/large/006tNbRwly1fya0g4603jj30x204mgr0.jpg)

  - BUILD文件由一系列编译目标组成，编译的先后顺序没有关系

    - 每一个编译目标第一行指定编译方式：库函数用library，主函数用binary
    - 编译目标的主体需要很多信息：name(编译目标的名字), src(源代码), deps(依赖关系)

    - ![屏幕快照 2018-12-17 下午8.23.40](https://ws2.sinaimg.cn/large/006tNbRwly1fya0gz5b8ej30ww0bgtmp.jpg)

      ![屏幕快照 2018-12-17 下午8.25.14](https://ws4.sinaimg.cn/large/006tNbRwly1fya0ivnb4mj30x4050ag2.jpg)

  - 执行 bazel build :hello_main，得到下面的结果，这些结果都是以软连接的方式存在当前的项目空间里，实际的编译结果都会保存到～/.cache/bazel目录下

    - ![屏幕快照 2018-12-17 下午8.35.01](https://ws3.sinaimg.cn/large/006tNbRwly1fya0sox1hxj30y807cte7.jpg)

  - 其中当前文件夹内的bazel-bin内存放了编译产生的二进制文件，所以执行

    `bazel-bin/hello_main` 就可以输出“Hello world”

## 2.2 tensorflow的安装

就不多记录了