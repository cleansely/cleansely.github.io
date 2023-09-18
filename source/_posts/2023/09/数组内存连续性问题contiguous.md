---
title: 数组内存连续性问题contiguous
date: 2023-09-12 22:58:57
tags:
categories: 昇腾
---
# 数组内存连续性问题contiguous

## 起因

对于一张图片使用numpy读取，进行了一定处理后作为输入1输入到onnx或者om进行推理，得到结果1，对于输入1保存为npy再读取作为输入2，输入到onnx或om进行推理，得到结果2，结果1和结果2并不相等。

![mmexport1694530147808](https://cleansely.top:48083/i/2023/09/12/65007b9c241ff.png)

## 解决

变量可以通过重新开辟空间，将数据连续拷贝进去的方法将不连续的数据变成某种连续方式。numpy 变量中连续性可以用自带的函数修正，不连续的变量通过函数 `np.ascontiguousarray(arr)`变为C连续，`np.asfortranarray(arr)`变为Fortran连续

```python
import numpy as np


if __name__ == '__main__':
    arr = np.arange(12).reshape(3, 2, 2)
    print(arr.data.c_contiguous)  # True arr C连续

    tran_arr = arr.transpose(2, 0, 1)
    print(tran_arr.data.contiguous)  # False tran_arr不连续

    c_arr = np.ascontiguousarray(tran_arr)  # 变为 C 连续
    print(c_arr.flags)

    f_arr = np.asfortranarray(tran_arr)  # 变为 Fortran 连续
    print(f_arr.flags)

    pass

"""
False
False
True
  C_CONTIGUOUS : True
  F_CONTIGUOUS : False
  OWNDATA : False
  WRITEABLE : True
  ALIGNED : True
  WRITEBACKIFCOPY : False
  UPDATEIFCOPY : False
"""
```



## 参考

https://www.zywvvd.com/notes/study/deep-learning/numpy-tensor-contiguous/numpy-tensor-contiguous/