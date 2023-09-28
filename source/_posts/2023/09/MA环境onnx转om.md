---
title: MA环境onnx转om
date: 2023-09-28 16:00:08
tags: 
    - modelarts 
    - onnx
    - om
categories: 昇腾
---

# MA环境onnx转om

## 参考

[Resnet50-推理](https://gitee.com/ascend/ModelZoo-PyTorch/blob/master/ACL_PyTorch/built-in/cv/Resnet50_Pytorch_Infer/README.md#https://gitee.com/link?target=https%253A%252F%252Fdownload.pytorch.org%252Fmodels%252Fresnet50-0676ba61.pth)

## pth转onnx

### 安装依赖
```
pip install torch torchvision
```

### pth转onnx

```
import sys

import torch
import torch.onnx
import torchvision.models as models

def convert(pthfile):
    model = models.resnet50(pretrained=False)
    resnet50 = torch.load(pthfile, map_location='cpu')
    model.load_state_dict(resnet50)
    print(model)

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    dummy_input = torch.randn(16, 3, 224, 224)
    torch.onnx.export(
        model, 
        dummy_input,
        "resnet50_official.onnx",
        input_names=input_names, 
        output_names=output_names, 
        opset_version=11)

pthfile = 'resnet50-0676ba61.pth' 
convert(pthfile)
```

## onnx转om

### 安装依赖

```
pip install onnxruntime
```

### 获取输入输出信息

```
from pprint import pprint
import onnxruntime

onnx_path = "resnet50_official.onnx"

provider = "CPUExecutionProvider"
onnx_session = onnxruntime.InferenceSession(onnx_path, providers=[provider])

print("----------------- 输入部分 -----------------")
input_tensors = onnx_session.get_inputs()  # 该 API 会返回列表
for input_tensor in input_tensors:         # 因为可能有多个输入，所以为列表
    
    input_info = {
        "name" : input_tensor.name,
        "type" : input_tensor.type,
        "shape": input_tensor.shape,
    }
    pprint(input_info)

print("----------------- 输出部分 -----------------")
output_tensors = onnx_session.get_outputs()  # 该 API 会返回列表
for output_tensor in output_tensors:         # 因为可能有多个输出，所以为列表
    
    output_info = {
        "name" : output_tensor.name,
        "type" : output_tensor.type,
        "shape": output_tensor.shape,
    }
    pprint(output_info)
    

## 输出
'''
(mindspore_py39) [ma-user onnx2om]$python get_model_info.py 
----------------- 输入部分 -----------------
{'name': 'actual_input_1', 'shape': [16, 3, 224, 224], 'type': 'tensor(float)'}
----------------- 输出部分 -----------------
{'name': 'output1', 'shape': [16, 1000], 'type': 'tensor(float)'}
'''
```

### 转换权重

根据输入信息可知，模型输入一个name为actual_input_1的变量，其shape是[16, 3, 224, 224],转换得到resnet50_bs16.om

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
atc --model=resnet50_official.onnx --framework=5 --output=resnet50_bs16 --input_format=NCHW --input_shape="actual_input_1:16,3,224,224" --enable_small_channel=1 --log=error --soc_version=Ascend910A 
```

## om推理测试

### 安装依赖

https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench

```
(mindspore_py39) [ma-user onnx2om]$pip install ./*.whl
Looking in indexes: http://pip.modelarts.private.com:8888/repository/pypi/simple
Processing ./aclruntime-0.0.2-cp39-cp39-linux_aarch64.whl
Processing ./ais_bench-0.0.2-py3-none-any.whl
Requirement already satisfied: numpy in /home/ma-user/anaconda3/envs/mindspore_py39/lib/python3.9/site-packages (from ais-bench==0.0.2) (1.25.0)
Requirement already satisfied: tqdm in /home/ma-user/anaconda3/envs/mindspore_py39/lib/python3.9/site-packages (from ais-bench==0.0.2) (4.65.0)
Installing collected packages: ais-bench, aclruntime
Successfully installed aclruntime-0.0.2 ais-bench-0.0.2
```

### om信息打印

```
import sys

import aclruntime
import numpy as np

def get_model_info():
    device_id = 0
    options = aclruntime.session_options()
    # 方法1设置级别为debug模式后可以打印模型信息
    options.log_level = 1
    session = aclruntime.InferenceSession(model_path, device_id, options)

    # 方法2 直接打印session 也可以获取模型信息
    print(session)

    # 方法3 也可以直接通过get接口去获取
    intensors_desc = session.get_inputs()
    for i, info in enumerate(intensors_desc):
        print("input info i:{} shape:{} type:{} val:{} realsize:{} size:{}".format(
            i, info.shape, info.datatype, int(info.datatype), info.realsize, info.size))

    intensors_desc = session.get_outputs()
    for i, info in enumerate(intensors_desc):
        print("outputs info i:{} shape:{} type:{} val:{} realsize:{} size:{}".format(
            i, info.shape, info.datatype, int(info.datatype), info.realsize, info.size))

         
device_id = 0
model_path = './resnet50_bs16.om'
get_model_info()   
 
```


### om推理示例


```
import sys

import aclruntime
import numpy as np

model_path = './resnet50_bs16.om'

# 最短运行样例
def infer_simple():
    device_id = 0
    options = aclruntime.session_options()
    session = aclruntime.InferenceSession(model_path, device_id, options)

    # create new numpy data according inputs info
    barray = bytearray(session.get_inputs()[0].realsize)
    ndata = np.frombuffer(barray)
    # convert numpy to pytensors in device
    tensor = aclruntime.Tensor(ndata)
    tensor.to_device(device_id)

    outnames = [ session.get_outputs()[0].name ]
    feeds = { session.get_inputs()[0].name : tensor}

    outputs = session.run(outnames, feeds)
    print("outputs:", outputs)

    outarray = []
    for out in outputs:
        # convert acltenor to host memory
        out.to_host()
        # convert acltensor to numpy array
        outarray.append(np.array(out))
    # summary inference throughput
    print("infer avg:{} ms".format(np.mean(session.sumary().exec_time_list)))
    
infer_simple()  
```

## 输出结果

### onnx结果

```
import onnxruntime
import numpy as np

sess = onnxruntime.InferenceSession('resnet50_official.onnx', providers=[ 'CPUExecutionProvider',])
a = np.ones((16,3, 224, 224)).astype(np.float32)

output = sess.run(['output1'], {'actual_input_1': a,})[0]

print(output)
```

```
[[-0.30805793  0.07984409 -1.1900382  ... -1.6530751   0.11777535
   0.24357179]
 [-0.30805793  0.07984409 -1.1900382  ... -1.6530751   0.11777535
   0.24357179]
 [-0.30805793  0.07984409 -1.1900382  ... -1.6530751   0.11777535
   0.24357179]
 ...
 [-0.30805793  0.07984409 -1.1900382  ... -1.6530751   0.11777535
   0.24357179]
 [-0.30805793  0.07984409 -1.1900382  ... -1.6530751   0.11777535
   0.24357179]
 [-0.30805793  0.07984409 -1.1900382  ... -1.6530751   0.11777535
   0.24357179]]
```


### om结果

```
import sys

import aclruntime
import numpy as np

model_path = './resnet50_bs16.om'

# 最短运行样例
def infer_simple():
    device_id = 0
    options = aclruntime.session_options()
    session = aclruntime.InferenceSession(model_path, device_id, options)

    # create new numpy data according inputs info
    barray = bytearray(session.get_inputs()[0].realsize)
    ndata = np.frombuffer(barray)
    # convert numpy to pytensors in device
    a = np.ones((64,3, 224, 224)).astype(np.float32)
    tensor = aclruntime.Tensor(a)
    tensor.to_device(device_id)

    outnames = [ session.get_outputs()[0].name ]
    feeds = { session.get_inputs()[0].name : tensor}

    outputs = session.run(outnames, feeds)
    print("outputs:", outputs)

    outarray = []
    for out in outputs:
        # convert acltenor to host memory
        out.to_host()
        # convert acltensor to numpy array
        outarray.append(np.array(out))
    print(outarray)
    # summary inference throughput
    print("infer avg:{} ms".format(np.mean(session.sumary().exec_time_list)))
    
infer_simple()  


```

```
INFO] acl init success
[INFO] open device 0 success
[INFO] load model ./resnet50_bs16.om success
[INFO] create model description success
outputs: [<Tensor>
shape:  (16, 1000)
dtype:  float32
device: 0]
[array([[-0.30737305,  0.0791626 , -1.1875    , ..., -1.6552734 ,
         0.11328125,  0.24133301],
       [-0.30737305,  0.0791626 , -1.1875    , ..., -1.6552734 ,
         0.11328125,  0.24133301],
       [-0.30737305,  0.0791626 , -1.1875    , ..., -1.6552734 ,
         0.11328125,  0.24133301],
       ...,
       [-0.30737305,  0.0791626 , -1.1875    , ..., -1.6552734 ,
         0.11328125,  0.24133301],
       [-0.30737305,  0.0791626 , -1.1875    , ..., -1.6552734 ,
         0.11328125,  0.24133301],
       [-0.30737305,  0.0791626 , -1.1875    , ..., -1.6552734 ,
         0.11328125,  0.24133301]], dtype=float32)]
infer avg:2.000999927520752 ms
[INFO] unload model success, model Id is 1
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
```

## 精度对比

对比完成，后续再更新

### 下载工具

```
git clone https://gitee.com/ascend/ait.git
```

### 运行对比工具

```
cd ait/ait/
chmod u+x install.sh
./install.sh
cd /home/ma-user/work/202309/onnx2om/ait/ait/components/debug/compare
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 main.py -m /home/ma-user/work/202309/onnx2om/resnet50_official.onnx -om /home/ma-user/work/202309/onnx2om/resnet50_bs16.om \
-c /usr/local/Ascend/ascend-toolkit/latest -o /home/ma-user/work/202309/onnx2om/result/test
```

## 优化