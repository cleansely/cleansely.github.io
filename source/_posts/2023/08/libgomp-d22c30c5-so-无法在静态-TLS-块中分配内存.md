---
title: libgomp-d22c30c5.so 无法在静态 TLS 块中分配内存
date: 2023-08-15 13:16:38
tags: 
    - modelarts 
categories: 昇腾
---

# libgomp-d22c30c5.so 无法在静态 TLS 块中分配内存

## 问题

libgomp-d22c30c5.so 无法在静态 TLS 块中分配内存
libgomp-d22c30c5.so.1.0.0:cannot allocate memory in static TLS block

## 解决

解决方法：在环境变量LD_PRELOAD中添加 libgomp-d22c30c5.so 路径


```
pip install scikit-learn
```

```
find / -name libgomp-d22c30c5.so.1.0.0
```

```
export LD_PRELOAD=$LD_PRELOAD:<绝对路径>/libgomp-d22c30c5.so.1.0.0
```