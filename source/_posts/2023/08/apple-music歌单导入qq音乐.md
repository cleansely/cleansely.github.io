---
title: apple music歌单导入qq音乐
date: 2023-08-12 21:55:07
tags: python
categories: 折腾日记
---

听歌软件从apple music换到了qq音乐，使用的是安卓手机，歌单的导入没找到合适的方式，就开始折腾了



## 思路

在apple music的app上分享歌单获取web网址，查看源码获取歌单的html信息，解析出歌名



## 记录



### 分享歌单

分享歌单到微信等软件，复制web地址到游览器打开

![Screenshot_20230812_220140](http://cleansely.top:48083/i/2023/08/12/64d791766d46f.jpg)



### 找到歌单相关信息

右键查看网页源码，搜索某个歌名找到对应歌单列表的信息

![image-20230812220508063](http://cleansely.top:48083/i/2023/08/12/64d79198cb492.png)

![image-20230812220646193](http://cleansely.top:48083/i/2023/08/12/64d791fcd10a4.png)

### 代码解析

很规范的字典列表嵌套，直接输出即可

```python
a = "..."

for i in a['track']:
    print(i['name'].split("(")[0])
```



![image-20230812220843281](http://cleansely.top:48083/i/2023/08/12/64d7926f69b36.png)



### qq音乐导入

之前可以成功导入，这次导入不行，不知道发生了什么

![Screenshot_20230812_221037](http://cleansely.top:48083/i/2023/08/12/64d7938872f5a.jpg)

![Screenshot_20230812_221255](http://cleansely.top:48083/i/2023/08/12/64d79392df9a7.jpg)
