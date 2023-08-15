---
title: notebook拷贝到obs
date: 2023-08-15 13:05:28
tags: 
    - modelarts 
    - obs
categories: 昇腾
---

# notebook拷贝到obs

## 文件拷贝

文件可以从demo.py重命名成test.py

```python
import moxing as mox
mox.file.copy('./demo.py', 'obs://path/test.py')
```

## 文件夹拷贝

文件夹test_dir会被重命名成test_path

```python
import moxing as mox
mox.file.copy_parallel('./test_dir', 'obs://path/test_path')
```