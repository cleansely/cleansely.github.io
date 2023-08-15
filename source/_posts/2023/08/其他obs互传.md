---
title: 其他obs互传
date: 2023-08-15 13:10:55
tags: 
    - modelarts 
    - obs
categories: 昇腾
---

# 其他obs互传

```python
from modelarts.session import Session
session = Session(access_key='',secret_key='', project_id='cn-east-xxx', region_name='cn-east-xxx')

session.obs.download_dir(src_obs_dir="obs://lfx/vicuna-7b", dst_local_dir="./")
```