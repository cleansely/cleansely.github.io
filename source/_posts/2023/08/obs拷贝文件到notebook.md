---
title: obs拷贝文件到notebook
date: 2023-08-15 13:09:41
tags: 
    - modelarts 
    - obs
categories: 昇腾
---

# obs拷贝文件到notebook

## 文件拷贝
```python
from modelarts.session import Session

session = Session()
session.obs.download_file(src_obs_file="obs://path/model.zip", dst_local_dir="./")
```

## 文件夹拷贝
dir01目录会出现在./
```python
from modelarts.session import Session

session = Session()
session.obs.download_dir(src_obs_dir="obs://path/dir01/", dst_local_dir="./")
```

