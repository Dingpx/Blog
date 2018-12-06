---
title: Alfred入门
date: 2018-12-04 17:08:29
tags: [工具, Alfred, 入门]
---

[TOC]

一直看别人对于这个软件的赞赏，没来得及去真正去了解使用过，这篇旨在帮助自己入门Alfred，熟悉界面功能，并尝试添加常用的workflow，以后会不定期更新。

<!--more-->

## 安装

安装的网址就是官网即可，主要的是官网版无法提供workflow的操作，所以需要破解注册机，注册机已经上传到我的网盘（如何破解可以参考https://www.jianshu.com/p/7d92bcaf2d49）

## 配置Alfred

这篇blog写的很全，下面就摘录几条自己常用的快捷键操作，参考链接（https://www.cnblogs.com/baitongtong/p/8298456.html）

（https://sspai.com/post/32979这个链接讲的内容更多更细，可供参考）

- 打开：option + 空格

- 默认不需要把所有文件类型加入搜索里面，而**使用Find+空格+文件名来查询文件或文件夹；使用Open+空格+文件名也可以**

- 目前Alfred只可检索Safari的书签，若你想检索Chrome的书签。则需要将Chrome书签导入到Safari中。导入步骤为：**打开Safari -> 菜单【文件】 -> 【导入自】 -> 谷歌Chrome导入书签数据**。

- 可自行增加自己需要的搜索，下面是我常用的网址URL备份

  - 百度:    <http://www.baidu.com/s?ie=UTF-8&wd={query}>
  - 知乎：[http://www.zhihu.com/search?q={query}&type1=all](http://www.zhihu.com/search?q=%7Bquery%7D&type1=all)
  - 有道翻译：[http://dict.youdao.com/search?q={query}](http://dict.youdao.com/search?q=%7Bquery%7D)
  - GitHub: https://github.com/search?q={query}
  - groundai: https://github.com/search?q={query}
  - Arxiv: https://arxiv.org/search/?query={query}&searchtype=all&source=header

- snippet功能：
- 查找文件时，直接空格一下就好，然后按->即可以查找这个文件夹下的内容，同理向上查找也是一样

## 配置workflow

直接去网上下载自己需要的workflow就可以，目前我常用的是

**cdto**: 直接找到terminal上的文件，不用反复的cd，真滴很方便

**Terminalfinder**:可以在当前find界面打开terminal或者在terminal界面打开find,链接如下（http://www.packal.org/workflow/terminalfinder）

**packal**:一个比较多的workflow应用的集合http://www.packal.org/workflow/packal-workflow-search

Kill,relaunch,Uninstall:这些都可以在packal里面找到

## 编写自己需要的workflow

主要根据https://www.jianshu.com/p/4b980a0193b6上讲的内容，配置两款自己需要的workflow

### 自动更新发布hexo博客

- 点击左下角的+号 --> Blank Workflow，内容填写如下

![屏幕快照 2018-12-06 下午2.39.44](https://ws1.sinaimg.cn/large/006tNbRwly1fxx11lm7pyj30wm0kc43a.jpg)

- 添加后选中新建的Blank Workflow 右键input->keyword

  ![屏幕快照 2018-12-06 下午2.40.21](https://ws1.sinaimg.cn/large/006tNbRwly1fxx1288jz3j31b20omak6.jpg)

  ![屏幕快照 2018-12-06 下午2.41.20](https://ws2.sinaimg.cn/large/006tNbRwly1fxx12ubyl1j31bc0g0dmy.jpg)

  ![屏幕快照 2018-12-06 下午2.41.28](https://ws2.sinaimg.cn/large/006tNbRwly1fxx13cjzpij30e609o0tp.jpg)

- 配置好后再右上角 + Action->Terminal Command

  ![屏幕快照 2018-12-06 下午2.48.43](https://ws2.sinaimg.cn/large/006tNbRwly1fxx13nizepj30w00jqqfl.jpg)

  ![屏幕快照 2018-12-06 下午2.48.52](https://ws3.sinaimg.cn/large/006tNbRwly1fxx143fr13j31360octf1.jpg)

- 保存就完成了

  ![屏幕快照 2018-12-06 下午2.49.00](https://ws1.sinaimg.cn/large/006tNbRwly1fxx14icmtpj30p00bawg2.jpg)

### 自动激活anaconda环境并打开juypter notebook

根据以上的步骤走一遍就好

![屏幕快照 2018-12-06 下午3.03.25](https://ws3.sinaimg.cn/large/006tNbRwly1fxx1edhrnaj31b20rw7ax.jpg)

