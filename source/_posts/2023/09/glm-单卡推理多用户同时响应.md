---
title: glm 单卡推理多用户同时响应
date: 2023-09-18 12:49:18
tags: 
    - modelarts 
    - mindformers
    - chatglm
categories: 昇腾
---
# 一、**背景**

前期对接客户过程中，反馈存在进行单卡多用户并发相应的需求，但当前官方模型均只支持串行响应无法对多个客户进行同时并发响应，因此需要对已有的 glm 代码进行开发，使其满足多用户并行响应的需求。

# 二、**操作步骤**

## **2.1 准备工作**

- 硬件：Ascend 910A
- MindSpore：2.0.0rc1
- CANN 版本：6.3.RC1.alpha003
- 驱动：23.0.rc1

宁波 AICC 直接注册：swr.cn-east-xxx.nbaicc.com/nbaicc_pub/ms20rc1_py39_cann63:2023062601

使用的环境为 mindspore_py39，可使用如下命令进行环境切换

```
cd /home/ma-user/work
mkdir MindFormers
cd MindFormers
conda activate mindspore_py39
```

在运行路径下，拉取最新的 mindformers 源码并安装

```
git clone -b dev https://gitee.com/mindspore/mindformers.git
cd mindformersbash build.sh
cd ..
```

## **2.2 创建空间进行多用户数据存储**

为了保证推理效率，在进行推理过程中均会打开增量推理开关，即将网络配置文件中的 use_past 变量设置为 True，在该条件下除了第一次需要输入完整语句外，后续每一次运行时均只需要输入最新的一个 token，其余内容均以保存在网络中。

为了满足多用户并发推理的需求，有两种解决方案可供选择，其一是关闭 use_past 开关，即每次推理输入全量的语句以及新生成的内容，这种情况下，各个用户之间的推理互不干涉但由于需要计算的 token 长度大，推理速度是受到明显限制的。另一种方案则是保持 use_past 开关为开启状态，在网络中对各个用户的前序状态进行保存，该方法能保证在增量推理过程中每次只输入一个 token 保证运算效率，是一种以空间换时间的思路。

对 mindformers/models/glm/layers.py 文件中的 DeepNormWithGLULayer 类__init__函数中的 key_past 与 value_past 赋值函数：

```python
self.key_past = Parameter(Tensor(np.zeros(shape=self.kv_shape), self.params_dtype), name="key_past")
self.value_past = Parameter(Tensor(np.zeros(shape=self.kv_shape), self.params_dtype), name="value_past")
```

修改为：

```python
self.kv_past = nn.CellList()
for tmp in range(users):
    self.kv_past.append(userskv(self.kv_shape, self.params_dtype))
```

并在 DeepNormWithGLULayer 类前定义新类用于存储多用户的 kv 值：

```python
class userskv(nn.Cell):
    def __init__(self, kv_shape, params_dtype):
        super(userskv, self).__init__()
        self.k_past = Parameter(Tensor(np.zeros(shape=kv_shape), params_dtype))
        self.value_past = Parameter(Tensor(np.zeros(shape=kv_shape), params_dtype))
```

对 DeepNormWithGLULayer 类 layer_forward 函数中的 key_past 与 value_past 赋值函数进行修改：

```python
key_reset = self.assign(self.key_past, self.mul_4(self.key_past, F.cast(init_reset, self.params_dtype)))
value_reset = self.assign(self.value_past,
                                      self.mul_4(self.value_past, F.cast(init_reset, self.params_dtype)))
                                      
key_update = self.assign(self.key_past, key_present)
value_update = self.assign(self.value_past, value_present)
```

修改为：

```python
key_reset = self.assign(self.kv_past[user].k_past, self.mul_4(self.kv_past[user].k_past, F.cast(init_reset, self.params_dtype)))
value_reset = self.assign(self.kv_past[user].value_past,
                                      self.mul_4(self.kv_past[user].value_past, F.cast(init_reset, self.params_dtype)))
                                      
key_update = self.assign(self.kv_past[user].k_past, key_present)
value_update = self.assign(self.kv_past[user].value_past, value_present)
```

## **2.3 用户信息传递**

由于在网络中为多用户前序信息创建了存储空间，因此在网络运行过程中需要传递入对应的用户编号信息，保证能够获取对应用户的前序信息。

对 mindformers/models/glm/layers.py 文件中的 DeepNormWithGLULayer 类 layer_forward 函数中的函数定义增加 user 变量：

```python
def layer_forward(self, hidden_states, mask, position_ids, init_reset=True, batch_valid_length=None, user=0):
```

对 mindformers/models/glm/layers.py 文件中的 DeepNormWithGLULayer 类 construct 函数中的函数中增加 user 变量：

```python
def construct(self, hidden_states, mask, position_ids, init_reset=True, batch_valid_length=None, user=0):
        '''
            hidden_states: [batch, seq_len, hidden_size]
            mask: [(1, 1), seq_len, seq_len]
        '''
        return self.layer_forward(hidden_states, mask, position_ids, init_reset, batch_valid_length, user)
```

对 mindformers/models/glm/glm.py 文件中的 GLMModel 类 construct 函数中的函数中增加 user 变量：

```python
def construct(self, input_ids, position_ids, attention_mask, init_reset=True, batch_valid_length=None, user=0):


    for i in range(self.num_layers):
            layer_ret = self.layers[i](hidden_states, attention_mask, position_ids, init_reset, batch_valid_length, user)
```

对 mindformers/models/glm/glm.py 文件中的 GLMForPreTraining 类_incremental_infer 函数中的函数中增加 user 变量：

```python
def _incremental_infer(self, input_ids, current_index, valid_length_each_example, position_ids=None, attention_mask=None, user=0):
        if self.is_first_iteration:
            self.add_flags_recursive(is_first_iteration=True)
            res = self(
                input_ids=Tensor(input_ids, mstype.int32),
                # input_ids (1,512) int32
                position_ids=Tensor(position_ids, mstype.int32),
                # position_ids (1, 2, 512) int32
                attention_mask=Tensor(attention_mask, mstype.float32),
                # attention_mask (1, 1, 512, 512) float32
                input_position=current_index,
                init_reset=Tensor([False], mstype.bool_),  # init_reset (1,) bool False
                batch_valid_length=Tensor([valid_length_each_example], mstype.int32),
                user=user
            )  # batch_valid_length (1,) int32 4
            # first iter done, go to other iters
            self.is_first_iteration = False
        else:
            self.add_flags_recursive(is_first_iteration=False)
            current_index_tmp = int(current_index[0])
            # use numpy to slice array to avoid complie ascend slice op
            inputs_tmp = input_ids[:, current_index_tmp:current_index_tmp + 1]
            position_ids_tmp = position_ids[..., current_index_tmp:current_index_tmp + 1]
            attention_mask_tmp = attention_mask[:, :, current_index_tmp:current_index_tmp + 1, :]
            res = self(
                input_ids=Tensor(inputs_tmp, mstype.int32),
                # input_ids (1,512) int32
                position_ids=Tensor(position_ids_tmp, mstype.int32),
                # position_ids (1, 2, 1) int32
                attention_mask=Tensor(attention_mask_tmp, mstype.float32),
                # attention_mask (1, 1, 1, 512) float32
                input_position=current_index,
                init_reset=Tensor([True], mstype.bool_),  # init_reset (1,) bool True
                batch_valid_length=Tensor([valid_length_each_example], mstype.int32),
                user=user
            )  # batch_valid_length (1,) int32 5
```

对 mindformers/models/glm/glm.py 文件中的 GLMForPreTraining 类_stream_chat 函数中的函数中增加 user 变量：

```python
def _stream_chat(self,
                 origin_inputs,
                 top_k,
                 top_p,
                 repetition_penalty,
                 max_length,
                 eos_token_id,
                 streamer=None,
                 pad_token_id=None,
                 tokenizer=None,
                    tlock=None, qi=0):
                    
        res = self._incremental_infer(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        current_index=current_index,
                        valid_length_each_example=valid_length_each_example,
                        user=qi
                    )
```

## **2.4 多用户请求测试**

完成代码修改后可通过如下代码进行多用户同时请求的响应测试，同时可相应的用户数量可通过 users 数量配置进行设置，问题最大回答长度可通过配置 seq_length 参数进行配置。

运行代码后会首先对每一个用户对应的计算图进行构造，用户数越多，需要初始化加载的时间越多。

由于当前采用了内存空间换时间的推理策略，能够同时响应的用户数和最大响应长度存在关联关系，初步测试数：长度为 1024 时最大是 17 个用户，平均响应速度在 0.9tokens/s 左右，综合是 15.3tokens/s；长度为 2048 时能够跑通 10 个用户，响应速度是 1.36tokens/s，综合是 13.6tokens/s。

```python
import time
import mindspore as ms
import numpy as np
import argparse
from mindformers.models.glm import GLMConfig, GLMChatModel, GLMChatModelWithLora
from mindformers.models.glm.chatglm_6b_tokenizer import ChatGLMTokenizer
from mindformers.models.glm.glm_processor import process_response
from mindspore import Profiler

users = 10

config = GLMConfig(
    position_encoding_2d=True,
    use_past=True,
    is_npu_acceleration=True,
    users=users,
    seq_length=2048
)
ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=6)
# profiler = Profiler()
# chat_glm(1)
# profiler.analyse()
model = GLMChatModel(config)
import threading

tlock = threading.Lock()

ms.load_checkpoint("/home/ma-user/work/mindformers/checkpoint_download/glm/glm_6b.ckpt", model)
tokenizer = ChatGLMTokenizer('/home/ma-user/work/mindformers/checkpoint_download/glm/ice_text.model')

q = {}
for i in range(users):
    q[i] = "你是谁"

def chat_glm(qi):
    # q = {0:"今天天气不错",1:"请介绍一下华为",2:"你好",3:"怎么提升网球技术",4:"你是谁",5:"你能做什么",6:"笑一个？",7:"坐飞机需要注意什么"}
    
    inputs = tokenizer(q[qi])
    start_time = time.time()
    count = 0

    for i in model._stream_chat(
        origin_inputs=np.expand_dims(np.array(inputs['input_ids']).astype(np.int32), 0),
        top_k=1,
        top_p=1,
        repetition_penalty=1,
        max_length=51,
        eos_token_id=config.eos_token_id,
        pad_token_id=config.pad_token_id,
        tokenizer=tokenizer, tlock=tlock, qi=qi):
        print(qi,time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), i)
        count+=1
    end_time = time.time()
    # print(model.transformer.layers[9].key_past.numpy())
    # outputs[0] = outputs[0][len(inputs['input_ids']) + 1:]
    # print(origin_inputs)
    print(f'generate speed: {count/(end_time-start_time):.2f} tokens/s')
    #profiler.analyse()

#chat_glm(0)
# profiler.analyse()
    
for i in range(users):
    th = threading.Thread(target=chat_glm, args=(i,))
    th.start()
```

# 三、**问题及解决**

## 3.1、用户前序数据丢失问题

最初采用该思路进行推理时，将各个用户的前序数据存储于推理图外的 mindformers/models/glm/glm.py 文件中的 GLMForPreTraining 类_stream_chat 函数中，通过设置变量并通过等号进行赋值，但运行过程中发现出现如下图所示的用户历史记录信息混乱的问题。

![](https://cleansely.top:48083/i/2023/09/18/6507c8f8047f1.png)

后经定位发现仅采用等号进行赋值会出现仅记录了变量地址，跳出函数后相关变量被释放导致读不到历史数据的问题。因此后续尝试采用在 GLMForPreTraining 类中通过 Parameter 申请前序数据的存储内存空间，并通过 assign 操作对前序参数进行记录。

## 3.2、响应速度慢，多用户数据存储加载成为效率瓶颈

在解决问题一后，已经能够进行多用户的并发推理操作，但发现整体推理效率低，综合推理速度仅为 1-2token/s，定位后发现内存空间所在的函数不在计算图内，导致 assign 操作占用了绝大部分的推理时间，后续调整后将前序数据的存储过程放置到了能够生成计算图的 DeepNormWithGLULayer 类中，进而解决了效率问题。

# **四、总结**

基于本文的开发，可以实现单卡推理场景下的多用户并发响应。

但为了满足多用户的并发响应，需要针对每个用户进行不同计算图的构建，导致用户数多时需要花费较长的时间进行初始化才能够进行需求响应，同时并发数也需要根据需求进行初始配置，如果需要进行用户数调整则需要进行重新启动。

# 五、**参考资料**

[https://gitee.com/mindspore/mindformers/tree/dev](https://gitee.com/mindspore/mindformers/tree/dev)