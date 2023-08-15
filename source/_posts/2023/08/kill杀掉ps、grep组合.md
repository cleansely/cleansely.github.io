---
title: kill杀掉ps、grep组合
date: 2023-08-15 13:02:38
tags:
categories: 昇腾
---

# kill杀掉ps、grep组合

```
ps -ef |grep %s |grep -v grep |awk '{print $2}'|xargs kill -9
```