---
title: torch.multinomial相关报错
date: 2023-08-15 13:14:23
tags: 
    - modelarts 
categories: 昇腾
---

# torch.multinomial相关报错

该算子在老版本 CANN 包中存在 BUG，即使将其修改到cpu运行，也会报错。

## 报错相关语句

```python
next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
```

## 解决办法

```
import numpy as np
probs = probs.flatten().cpu().numpy().astype('float64')
sum_probs = np.absolute(probs).sum()
probs = probs/sum_probs
next_tokens = np.random.multinomial(1, probs)
next_tokens = next_tokens.argmax()
next_tokens = torch.tensor(next_tokens).npu()
```