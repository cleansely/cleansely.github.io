---
title: gelu_impl函数优化
date: 2023-08-15 13:17:13
tags: 
    - modelarts 
categories: 昇腾
---

# gelu_impl函数优化

## 原始代码

```python
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *                                       (1.0 + 0.044715 * x * x)))
```
## 修改

修改为：换成更快的fast_gelu

```python
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return torch.fast_gelu(x)
```