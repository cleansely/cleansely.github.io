---
title: ms-chatglm6b 单卡多用户流式推理临时方案
date: 2023-08-17 15:30:42
tags: 
    - modelarts 
    - mindformers
    - chatglm
categories: 昇腾
---

# ms-chatglm6b 单卡多用户流式推理临时方案

## 1. 拉取最新代码并安装

```
git clone https://gitee.com/mindspore/mindformers.git mindformers0816
cd mindformers0816
bash build.sh
```

## 2. 修改源码

可以修改当前路径代码，也可以修改环境安装代码`/home/ma-user/anaconda3/envs/mindspore_py39/lib/python3.9/site-packages/mindformers`

### 2.1 替换glm.py

```python
# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""GLM model."""
import os
import numpy as np

import mindspore as ms
from mindspore import dtype as mstype
from mindspore import nn, ops, Tensor
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter

from mindformers.mindformer_book import MindFormerBook
from mindformers.modules.transformer import VocabEmbedding, EmbeddingOpParallelConfig, OpParallelConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.core.loss import CrossEntropyLoss
from mindformers.modules.layers import LayerNorm
from mindformers.tools.utils import is_version_ge
from mindformers.pet.tuners.pet_adapter import PetAdapter
from mindformers.pet.tuners.lora_adapter import LoraAdapter

from .glm_config import GLMConfig
from .layers import DeepNormWithGLULayer
from ..base_model import BaseModel

#  Get MS backend: 0 vm 1 GE
is_ge = os.getenv('MS_ENABLE_GE')
if is_ge == '1':
    jit_level = "O3"
else:
    jit_level = "O1"

default_dpmp_config = OpParallelConfig()
default_embedding_parallel_config = EmbeddingOpParallelConfig()

__all__ = ['GLMForPreTraining', 'GLMChatModel', 'GLMForPreTrainingWithLora', 'GLMChatModelWithLora']


def topk_fun(logits, topk=5):
    """Get topk"""
    batch_value = []
    batch_index = []
    for i in range(logits.shape[0]):
        target_column = logits[i].tolist()
        sorted_array = [(k, v) for k, v in enumerate(target_column)]
        sorted_array.sort(key=lambda x: x[1], reverse=True)
        topk_array = sorted_array[:topk]
        index, value = zip(*topk_array)
        batch_value.append(value)
        batch_index.append(index)
    return np.array(batch_value), np.array(batch_index)


def batch_select(data, index):
    """bathc operation to sorted_logits[:, :top_p_num]"""
    output = []
    for i in range(data.shape[0]):
        res = data[i, :index[i]]
        output.append(res.reshape(1, -1))
    return np.concatenate(output, 0)


def sampler(log_probs_revised, top_p, top_k, use_pynative=False):
    """Convert the log_probs to probability"""
    if use_pynative:
        logits = P.Pow()(np.e, Tensor(log_probs_revised, mstype.float32))
    else:
        logits = np.power(np.e, np.array(log_probs_revised, np.float32))

    # If top_p is less than 1.0, use top_p sampling
    if top_p < 1.0:
        # Only consider the 5000 largest logits to reduce computation
        if use_pynative:
            sorted_logits, index = P.TopK(sorted=True)(logits, 5000)
            cumsum_logits = P.CumSum()(sorted_logits, 1)
            cumsum_logits = cumsum_logits.asnumpy()
            index = index.asnumpy()
            sorted_logits = sorted_logits.asnumpy()
        else:
            sorted_logits, index = topk_fun(logits, 5000)
            cumsum_logits = np.cumsum(sorted_logits, 1)
        cumsum_logits = cumsum_logits
        index = index
        sorted_logits = sorted_logits
        top_p_num = np.sum(cumsum_logits < top_p, axis=-1) + 1
        # Get the corresponding probs and indices
        probs = batch_select(sorted_logits, top_p_num)
        p_args = batch_select(index, top_p_num)
        p = probs / np.sum(probs, -1, keepdims=True)
        # if top_p is set to 1.0, use top_k sampling
    else:
        # Get the corresponding probs and indices
        if use_pynative:
            probs, p_args = P.TopK(sorted=True)(logits, top_k)
            probs = probs.asnumpy()
            p_args = p_args.asnumpy()
        else:
            probs, p_args = topk_fun(logits, top_k)
        probs = probs
        p_args = p_args
        # Avoid rounding error
        for i in range(probs.shape[0]):
            if np.sum(probs[i]) == 0:
                probs[i] = np.array([1 / top_k for _ in range(top_k)])
        p = probs / np.sum(probs, -1, keepdims=True)
    return p, p_args


def precision_correct(p, top_p, top_k, batch_size):
    # Avoid rounding error
    if top_p == 1:
        for i in range(batch_size):
            if np.sum(p[i]) == 0:
                p[i] = np.array([1 / top_k for _ in range(top_k)])
        p = p / np.sum(p, -1, keepdims=True)
    return p


class ProcessLogits(nn.Cell):
    r"""Process logits into probability distribution."""

    def __init__(self, use_past=False):
        super(ProcessLogits, self).__init__()
        self.e = ms.Tensor(np.e)
        self.gather = P.Gather()
        self.logsoftmax = P.LogSoftmax()
        self.reshape = P.Reshape()
        self.use_past = use_past

    def construct(self, logits, current_index=None, is_first_iteration=False):
        logits = logits.reshape(-1, logits.shape[-1])
        if self.use_past and not is_first_iteration:
            logits = logits
        elif current_index is not None:
            index = current_index.view(-1,)
            logits = self.gather(logits, index, 0)
        outputs = self.logsoftmax(logits)
        outputs = F.tensor_pow(self.e, outputs)
        return outputs


class GLMModel(nn.Cell):
    """
    The backbone of GLM network

    Args:
        config (GLMConfig): The config of network.
        op_parallel_config (optional): Operator parallel strategy. Default: `OpParallelConfig()`.
        embed_parallel_config (optional): Operator parallel strategy. Default: `EmbeddingOpParallelConfig()`.
    """
    def __init__(self,
                 config,
                 op_parallel_config=default_dpmp_config,
                 embed_parallel_config=default_embedding_parallel_config):
        super(GLMModel, self).__init__()
        # recording parameters
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.seq_length = config.seq_length
        self.use_past = config.use_past
        layernorm = LayerNorm
        if config.parallel_config:
            op_parallel_config = config.parallel_config

        # create embedding parameters
        if is_version_ge(ms.__version__, '1.11.0'):
            self.embedding_dropout = nn.Dropout(p=config.embedding_dropout_prob)
        else:
            self.embedding_dropout = nn.Dropout(keep_prob=1 - config.embedding_dropout_prob)

        embed_parallel_config.data_parallel = op_parallel_config.data_parallel
        embed_parallel_config.model_parallel = op_parallel_config.model_parallel
        embed_parallel_config.vocab_emb_dp = False
        self.word_embeddings = VocabEmbedding(vocab_size=config.vocab_size, embedding_size=config.hidden_size,
                                              parallel_config=embed_parallel_config)

        self.matmul = ops.MatMul().shard(((1, 1), (1, embed_parallel_config.model_parallel)))
        self.transpose = ops.Transpose().shard(((embed_parallel_config.model_parallel, 1),))

        def get_layer(layer_id):
            return DeepNormWithGLULayer(
                self.num_layers,
                self.hidden_size,
                self.num_heads,
                config.batch_size,
                config.attention_dropout_rate,
                config.hidden_dropout_rate,
                config.layernorm_epsilon,
                layer_id,
                max_seq_len=self.seq_length,
                inner_hidden_size=config.inner_hidden_size,
                hidden_size_per_attention_head=config.hidden_size_per_attention_head,
                layernorm_order=config.layernorm_order,
                layernorm=layernorm,
                use_bias=True,
                activation_func=config.activation_func,
                position_encoding_2d=config.position_encoding_2d,
                params_dtype=config.param_init_type,
                layernorm_dtype=config.layernorm_compute_type,
                softmax_dtype=config.softmax_compute_type,
                compute_dtype=config.compute_dtype,
                use_past=self.use_past,
                parallel_config=op_parallel_config,
                users=config.users,
            )

        self.layers = nn.CellList(
            [get_layer(layer_id) for layer_id in range(config.num_layers)])

        # Final layer norm before output.
        self.use_final_layernorm = config.use_final_layernorm
        if config.use_final_layernorm:
            self.final_layernorm = layernorm(config.hidden_size, eps=config.layernorm_epsilon)
            self.final_layernorm.shard(((op_parallel_config.data_parallel, 1, 1),))

    def construct(self, input_ids, position_ids, attention_mask, init_reset=True, batch_valid_length=None, user=0):
        """
        Get output logits

        Inputs:
            input_ids (Tensor): The tokenized inputs with dtype int32.
            input_mask (Tensor): The mask indicating whether each position is a valid input.
            position_ids (Tensor): Used to identify each token's position in the list of tokens.
            attention_mask (Tensor): Used when batching sequences together.
            init_reset (bool, optional): Default: True.
            batch_valid_length (Tensor, optional): Default: None.

        Returns:
            logits (Tensor): The output logit of backbone.
            table (Tensor): The embedding table for the vocabulary.
        """
        if attention_mask is None:
            attention_mask = ops.ones((1, 1), mstype.int32)

        hidden_states, table = self.word_embeddings(input_ids)

        hidden_states = self.embedding_dropout(hidden_states)

        for i in range(self.num_layers):
            layer_ret = self.layers[i](hidden_states, attention_mask, position_ids, init_reset, batch_valid_length, user)

            if isinstance(layer_ret, tuple):
                layer_ret = layer_ret[0]
            hidden_states = layer_ret

        # Final layer norm.
        if self.use_final_layernorm:
            logits = self.final_layernorm(hidden_states)
        else:
            logits = hidden_states

        return logits, table


class GLMHead(nn.Cell):
    r"""Head for GLM to get the logits of each token in the vocab."""

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 param_init_type=mstype.float32,
                 compute_dtype=mstype.float16,
                 embed_parallel_config=None):
        super(GLMHead, self).__init__()
        self.param_init_type = param_init_type
        self.compute_dtype = compute_dtype
        self.weight = Parameter(initializer("normal", [vocab_size, hidden_size], compute_dtype), name="weight")
        self.transpose = ops.Transpose().shard(((embed_parallel_config.model_parallel, 1),))
        self.matmul = ops.MatMul(transpose_b=True).shard(
            ((embed_parallel_config.data_parallel, 1), (embed_parallel_config.model_parallel, 1)))

    def construct(self, state, embedding_table=None):
        """Get vocab probs"""
        state = F.reshape(state, (-1, F.shape(state)[-1]))
        state = ops.cast(state, self.compute_dtype)
        if embedding_table is None:
            embedding_table = self.weight
        embedding_table = self.cast(embedding_table, self.compute_dtype)
        logits_parallel = self.matmul(state, embedding_table)
        return logits_parallel


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class GLMForPreTraining(BaseModel):
    r"""
    Provide glm training loss or logits through network.

    Args:
        config (GLMConfig): The config of GLMModel.
    """
    _support_list = MindFormerBook.get_model_support_list()['glm']

    def __init__(self, config: GLMConfig):
        super(GLMForPreTraining, self).__init__(config)
        self.config = config
        self.position_encoding_2d = config.position_encoding_2d
        self.transformer = GLMModel(config)
        self.lm_head = GLMHead(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            param_init_type=config.param_init_type,
            compute_dtype=config.compute_dtype,
            embed_parallel_config=config.parallel_config)
        self.stridedslice = ops.StridedSlice().shard(((1, 1),))
        self.loss = CrossEntropyLoss(parallel_config=config.parallel_config)
        self.gmask = config.gmask_token_id
        self.bos_token_id = config.bos_token_id
        self.ones = P.Ones()
        self.load_checkpoint(config)

    def get_masks_np(self, input_ids):
        batch_size, seq_length = input_ids.shape
        context_lengths = [list(seq).index(self.config.bos_token_id) for seq in input_ids]
        attention_mask = np.tril(np.ones((batch_size, seq_length, seq_length)))
        for i, context_length in enumerate(context_lengths):
            attention_mask[i, :, :context_length] = 1
        attention_mask = np.expand_dims(attention_mask, axis=1)
        attention_mask = np.array(attention_mask < 0.5, np.bool_)
        return attention_mask

    def get_position_ids_np(self, input_ids, mask_positions, use_gmasks=None):
        """Get position ids from input_ids and mask_positions with numpy"""
        batch_size, seq_length = input_ids.shape
        if use_gmasks is None:
            use_gmasks = [False] * batch_size
        context_lengths = [list(seq).index(self.config.bos_token_id) for seq in input_ids]
        if self.config.position_encoding_2d:
            position_ids = np.repeat(np.expand_dims(np.arange(seq_length), 0), batch_size, axis=0)
            for i, context_length in enumerate(context_lengths):
                position_ids[i, context_length:] = mask_positions[i]
            block_position_ids = [np.concatenate((
                np.zeros(context_length, np.int32),
                np.arange(seq_length - context_length, dtype=np.int32) + 1
            )) for context_length in context_lengths]
            block_position_ids = np.stack(block_position_ids, axis=0)
            position_ids = np.stack((position_ids, block_position_ids), axis=1)
        else:
            position_ids = np.repeat(np.expand_dims(np.arange(seq_length), 0), batch_size, axis=0)
            for i, context_length in enumerate(context_lengths):
                if not use_gmasks[i]:
                    position_ids[context_length:] = mask_positions[i]
        return position_ids

    def create_position_ids_np(self, input_ids):
        """Get position ids from input_ids with numpy"""
        mask, gmask = self.config.mask_token_id, self.config.gmask_token_id
        seqs = list(input_ids)

        mask_positions, use_gmasks = [], []
        for seq in seqs:
            mask_token = gmask if gmask in seq else mask
            use_gmask = mask_token == gmask
            mask_positions.append(list(seq).index(mask_token))
            use_gmasks.append(use_gmask)
        position_ids = self.get_position_ids_np(input_ids, mask_positions, use_gmasks=None)
        return position_ids

    def _incremental_infer(self,
                           input_ids,
                           current_index,
                           valid_length_each_example,
                           position_ids=None,
                           attention_mask=None,
                          user=0):
        # Claim the first graph
        print(user)
        if self.is_first_iteration:
            print("first=======")
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
            '''print("inputs_tmp", inputs_tmp)
            print("position_ids_tmp", position_ids_tmp)
            print("attention_mask_tmp", attention_mask_tmp)'''
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

        return res
    
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
        """
        Text generation given the model and origin inputs

        Inputs:
            model: The model to run the prediction
            end_token(int): The model will stop generating the words when it reaches the end_token.
            origin_inputs(list): The prompt for generation, should be a list of ids.
            model_origin_max_length(int): The sequence length of the model trained.
            max_length(int):  The maximum of generated length.
            vocab_size(int): The vocabulary length of the model.
            config: Inference configurations.
            streamer: Streamer object that will be used to stream the generated sequences.

        Returns:
            outputs: the ids for the generated text
        """
        if pad_token_id is None:
            pad_token_id = 0
        # Get configurations for inference
        use_pynative = True

        if streamer is not None:
            streamer.put(origin_inputs[0])

        batch_size = origin_inputs.shape[0]
        is_npu_acceleration = self.config.is_npu_acceleration
        valid_length_each_example = []
        for i in range(batch_size):
            # As the nonzero returns the index and we need length
            valid_length_each_example.append(np.max(np.argwhere(origin_inputs[i] != pad_token_id)) + 1)
        valid_length_each_example = np.array(valid_length_each_example)
        ori_valid_length_each_example = np.array(valid_length_each_example)
        if np.max(valid_length_each_example) > max_length:
            raise ValueError("The max_length set is smaller than the length in the input_ids. You shout set "
                             f"max_length to {np.max(valid_length_each_example)}")
        target_length = self.config.seq_length if max_length > self.config.seq_length else max_length
        # A list of the frequency of each token
        frequency_list = None
        input_ids = self._pad_inputs_using_max_length(origin_inputs=origin_inputs, pad_token_id=pad_token_id)

        # for GLM `attention_mask` and `position_ids` generation
        attention_mask = self.get_masks_np(input_ids)
        position_ids = self.create_position_ids_np(input_ids)

        input_mask = np.zeros_like(input_ids)
        for i in range(valid_length_each_example.shape[0]):
            input_mask[i, :valid_length_each_example[i]] = 1

        # A single loop generates one token, loop until reaching target model_origin_max_length or generating eod token
        is_finished = [False] * batch_size

        # setup is_first_iteration flag for incremental infer
        if self.config.use_past:
            self.is_first_iteration = True
        
        
        
        is_first_iteration = True
        pre_text = ''
        assign = ops.Assign()
        while np.sum(is_finished) != batch_size:
            tlock.acquire()
            # for GLM generation
            # model basic setting
            self.top_p = top_p
            self.top_k = top_k
            self.repetition_penalty = repetition_penalty

            seq_length = input_ids.shape[1]
            current_index = [valid_length_each_example[i] - 1 + i * seq_length for i in range(batch_size)]
            current_index = Tensor(current_index, mstype.int32)

            if self.config.use_past:
                self.is_first_iteration = is_first_iteration
                print(origin_inputs[0], is_first_iteration,self.is_first_iteration)
                # print(origin_inputs[0], id(input_ids))
                if is_first_iteration:
                    res = self._incremental_infer(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        current_index=current_index,
                        valid_length_each_example=valid_length_each_example,
                        user=qi
                    )
                    is_first_iteration = False
                    self.is_first_iteration = is_first_iteration
                    
                else:
                    
        
                    res = self._incremental_infer(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        current_index=current_index,
                        valid_length_each_example=valid_length_each_example,
                        user=qi
                    )
                    
                    
            else:
                res = self(
                    input_ids=Tensor(input_ids, mstype.int32),
                    position_ids=Tensor(position_ids, mstype.int32),
                    attention_mask=Tensor(attention_mask, mstype.float32)
                )
            if is_npu_acceleration:
                p, p_args = res
                p = p.asnumpy()
                p_args = p_args.asnumpy()
                # Avoid rounding error
                p = precision_correct(p, top_p, top_k, batch_size)
                #P.Depend()(p, tmp)
            else:
                log_probs = self.process_logits(res, current_index, is_first_iteration, self.config.use_past)
                # Sample
                log_probs = log_probs.asnumpy()
                vocab_size = log_probs.shape[-1]
                if repetition_penalty != 1 and frequency_list is None:
                    frequency_list = np.array([[0 for _ in range(vocab_size)]])
                log_probs_revised = log_probs.reshape(batch_size, vocab_size)
                if repetition_penalty != 1:
                    log_probs_revised = log_probs - frequency_list * repetition_penalty - \
                                        (frequency_list > 0) * repetition_penalty
                p, p_args = sampler(log_probs_revised, top_p, top_k, use_pynative)
            
            # Random select a token as final output for this round
            for i in range(batch_size):
                if is_finished[i]:
                    continue
                target_index = np.random.choice(len(p[i]), p=p[i])

                # update frequency list
                target = p_args[i][target_index]

                if repetition_penalty != 1:
                    frequency_list[0][target] = frequency_list[0][target] + 1
                input_ids[i, valid_length_each_example[i]] = p_args[i, target_index]

                if streamer is not None:
                    streamer.put(np.asarray([target]))

                valid_length_each_example[i] += int(1)
                input_mask[i][valid_length_each_example[i] - 1] = 1

                # Stop judgment
                if p_args[i][target_index] == eos_token_id or valid_length_each_example[i] == target_length:
                    is_finished[i] = True
                    continue
                else:
                    text_tmp = tokenizer.decode(
                        input_ids[i, int(ori_valid_length_each_example[i]):valid_length_each_example[i]])
                    if "�" in text_tmp:
                        continue
                    else:
                        cur_text = text_tmp
                        yield cur_text
            tlock.release()
            

    def _forward(self,
                 origin_inputs,
                 top_k,
                 top_p,
                 repetition_penalty,
                 max_length,
                 eos_token_id,
                 streamer=None,
                 pad_token_id=None):
        """
        Text generation given the model and origin inputs

        Inputs:
            model: The model to run the prediction
            end_token(int): The model will stop generating the words when it reaches the end_token.
            origin_inputs(list): The prompt for generation, should be a list of ids.
            model_origin_max_length(int): The sequence length of the model trained.
            max_length(int):  The maximum of generated length.
            vocab_size(int): The vocabulary length of the model.
            config: Inference configurations.
            streamer: Streamer object that will be used to stream the generated sequences.

        Returns:
            outputs: the ids for the generated text
        """
        if pad_token_id is None:
            pad_token_id = 0
        # Get configurations for inference
        use_pynative = True

        if streamer is not None:
            streamer.put(origin_inputs[0])

        batch_size = origin_inputs.shape[0]
        is_npu_acceleration = self.config.is_npu_acceleration
        valid_length_each_example = []
        for i in range(batch_size):
            # As the nonzero returns the index and we need length
            valid_length_each_example.append(np.max(np.argwhere(origin_inputs[i] != pad_token_id)) + 1)
        valid_length_each_example = np.array(valid_length_each_example)
        if np.max(valid_length_each_example) > max_length:
            raise ValueError("The max_length set is smaller than the length in the input_ids. You shout set "
                             f"max_length to {np.max(valid_length_each_example)}")
        target_length = self.config.seq_length if max_length > self.config.seq_length else max_length
        # A list of the frequency of each token
        frequency_list = None
        input_ids = self._pad_inputs_using_max_length(origin_inputs=origin_inputs, pad_token_id=pad_token_id)

        # for GLM `attention_mask` and `position_ids` generation
        attention_mask = self.get_masks_np(input_ids)
        position_ids = self.create_position_ids_np(input_ids)

        input_mask = np.zeros_like(input_ids)
        for i in range(valid_length_each_example.shape[0]):
            input_mask[i, :valid_length_each_example[i]] = 1

        # A single loop generates one token, loop until reaching target model_origin_max_length or generating eod token
        is_finished = [False] * batch_size

        # setup is_first_iteration flag for incremental infer
        if self.config.use_past:
            self.is_first_iteration = True

        is_first_iteration = False
        while np.sum(is_finished) != batch_size:
            # for GLM generation
            # model basic setting
            self.top_p = top_p
            self.top_k = top_k
            self.repetition_penalty = repetition_penalty

            seq_length = input_ids.shape[1]
            current_index = [valid_length_each_example[i] - 1 + i * seq_length for i in range(batch_size)]
            current_index = Tensor(current_index, mstype.int32)

            if self.config.use_past:
                is_first_iteration = self.is_first_iteration
                res = self._incremental_infer(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    current_index=current_index,
                    valid_length_each_example=valid_length_each_example
                )
            else:
                res = self(
                    input_ids=Tensor(input_ids, mstype.int32),
                    position_ids=Tensor(position_ids, mstype.int32),
                    attention_mask=Tensor(attention_mask, mstype.float32)
                )
            if is_npu_acceleration:
                p, p_args = res
                p = p.asnumpy()
                p_args = p_args.asnumpy()
                # Avoid rounding error
                p = precision_correct(p, top_p, top_k, batch_size)
            else:
                log_probs = self.process_logits(res, current_index, is_first_iteration, self.config.use_past)
                # Sample
                log_probs = log_probs.asnumpy()
                vocab_size = log_probs.shape[-1]
                if repetition_penalty != 1 and frequency_list is None:
                    frequency_list = np.array([[0 for _ in range(vocab_size)]])
                log_probs_revised = log_probs.reshape(batch_size, vocab_size)
                if repetition_penalty != 1:
                    log_probs_revised = log_probs - frequency_list * repetition_penalty - \
                                        (frequency_list > 0) * repetition_penalty
                p, p_args = sampler(log_probs_revised, top_p, top_k, use_pynative)

            # Random select a token as final output for this round
            for i in range(batch_size):
                if is_finished[i]:
                    continue
                target_index = np.random.choice(len(p[i]), p=p[i])

                # update frequency list
                target = p_args[i][target_index]

                if repetition_penalty != 1:
                    frequency_list[0][target] = frequency_list[0][target] + 1
                input_ids[i, valid_length_each_example[i]] = p_args[i, target_index]

                if streamer is not None:
                    streamer.put(np.asarray([target]))

                valid_length_each_example[i] += int(1)
                input_mask[i][valid_length_each_example[i] - 1] = 1

                # Stop judgment
                if p_args[i][target_index] == eos_token_id or valid_length_each_example[i] == target_length:
                    is_finished[i] = True
                    continue

        # Return valid outputs out of padded outputs
        output_ids = []
        for i in range(batch_size):
            output_ids.append(input_ids[i, : int(valid_length_each_example[i])].astype(np.int32))
        if streamer is not None:
            streamer.end()
        return output_ids

    # pylint: disable=W0613
    def construct(self, input_ids, label=None, position_ids=None, attention_mask=None,
                  input_position=None, init_reset=True, batch_valid_length=None, user=0):
        """
        Extract logits and calculate loss

        Inputs:
            input_ids (Tensor): The tokenized inputs with dtype int32.
            label (Tensor): The indices of input sequence tokens in the vocabulary.
            position_ids (Tensor): Used to identify each token's position in the list of tokens.
            attention_mask (Tensor): Used when batching sequences together.
            init_reset (bool, optional): Default: True.
            batch_valid_length(Tensor, optional): Default: None.

        Returns:
            Training phase:
                loss: Training loss.
            Other phase:
                logits (Tensor): The output logit of backbone.
        """
        batch_size, seq_length = input_ids.shape

        if self.phase == "train":
            tokens = self.stridedslice(input_ids, (0, 0), (batch_size, seq_length), (1, 1))
        else:
            tokens = input_ids

        output_states, _ = self.transformer(tokens, position_ids,
                                            attention_mask, init_reset, batch_valid_length, user)
        logits = self.lm_head(output_states)

        if self.phase != 'train':
            return logits

        logits_shape = logits.shape
        label = label.reshape((-1,))
        logits = logits.reshape((-1, logits_shape[-1]))
        input_mask = self.ones(tokens.shape, logits.dtype)
        input_mask = input_mask.reshape((-1,))
        loss = self.loss(logits, label, input_mask)
        return loss


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class GLMChatModel(GLMForPreTraining):
    r"""
    Provide glm chat capability through network.
    Args:
        config (GLMConfig): The config of GLMModel.

    Returns:
        Tensor, the probability distribution of network loss.
    """
    _support_list = MindFormerBook.get_model_support_list()['glm']

    def __init__(self, config: GLMConfig):
        super(GLMChatModel, self).__init__(config)
        self.e = ms.Tensor(np.e, dtype=mstype.float32)
        self.pow = P.Pow()
        self.topk = P.TopK(sorted=True)
        self.cumsum = P.CumSum()
        if is_version_ge(ms.__version__, '1.11.0'):
            self.sum = ops.sum
        else:
            self.sum = P.ReduceSum(keep_dims=False)
        self.vocab_size = config.vocab_size
        self.batch_size = config.batch_size
        self.frequency_list = ms.Tensor([[0 for _ in range(self.vocab_size)]])
        self.post_logits = ProcessLogits(use_past=config.use_past)
        # seems not supported yet.
        # self.top_p = config.top_p
        self.top_p = 1
        self.top_k = config.top_k
        self.repetition_penalty = config.repetition_penalty
        self.is_first_iteration = False
        self.is_npu_acceleration = config.is_npu_acceleration

    def sample(self, log_probs):
        """Convert the log_probs to probability"""

        if self.repetition_penalty != 1:
            log_probs = log_probs - self.frequency_list * self.repetition_penalty - \
                        (self.frequency_list > 0) * self.repetition_penalty

        # Process sample in graph to accelerate generate
        logits = self.pow(self.e, log_probs)

        # If top_p is less than 1.0, use top_p sampling
        # seems not supported yet.
        if self.top_p < 1.0:
            sorted_logits, index = self.topk(logits, 5000)
            cumsum_logits = self.cumsum(sorted_logits, 1)
            top_p_num = self.sum((cumsum_logits < self.top_p).astype(mstype.int32), -1) + 1
            top_p_num = int(top_p_num)
            # Get the corresponding probs and indices
            probs = sorted_logits[:, :top_p_num]
            p_args = index[:, :top_p_num]
            p = probs / self.sum(probs, -1, keepdim=True)

        # if top_p is set to 1.0, use top_k sampling
        else:
            probs, p_args = self.topk(logits, self.top_k)
            p = probs

        return p, p_args

    # pylint:disable=arguments-differ
    def construct(self, input_ids, position_ids=None, attention_mask=None,
                  input_position=None, init_reset=True, batch_valid_length=None, user=0):
        """Get probs and p_args"""
        # model forward
        output_states, _ = self.transformer(input_ids, position_ids, attention_mask, init_reset, batch_valid_length, user)
        logits = self.lm_head(output_states)

        if not self.is_npu_acceleration:
            return logits

        # logit post process
        log_probs = self.post_logits(logits, input_position, self.is_first_iteration)

        # logit sort and sample
        probs, p_args = self.sample(log_probs)

        return probs, p_args


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class GLMForPreTrainingWithLora(GLMForPreTraining):
    """GLM Model for pretraining with LoRA

    Args:
        config (GLMConfig): The config of network.
    """

    def __init__(self, config: GLMConfig = None, pet=None, **kwargs):
        _ = kwargs
        super().__init__(config)
        # get Pet tuning model.
        self.pet = pet
        self.pet.pet_config.reg_rules = r'.*query_key_value*'
        self.transformer = LoraAdapter.get_pet_model(self.transformer, self.pet.pet_config)
        # freeze pretrained model
        PetAdapter.freeze_pretrained_model(self, self.pet.pet_type)


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class GLMChatModelWithLora(GLMChatModel):
    """GLM Model for pretraining with LoRA

    Args:
        config (GLMConfig): The config of network.
    """

    def __init__(self, config: GLMConfig = None, pet=None, **kwargs):
        _ = kwargs
        ckpt_cfg = config.checkpoint_name_or_path
        config.checkpoint_name_or_path = None
        super().__init__(config)
        # get Pet tuning model.
        self.pet = pet
        self.pet.pet_config.reg_rules = r'.*query_key_value*'
        self.transformer = LoraAdapter.get_pet_model(self.transformer, self.pet.pet_config)
        config.checkpoint_name_or_path = ckpt_cfg
        self.load_checkpoint(config)
```

### 2.2 替换layers.py

```python
"""base transformer layer."""
import numpy as np

import mindspore as ms
from mindspore import Parameter, Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore import ops

from mindformers.modules.layers import LayerNorm, Linear
from mindformers.tools.utils import is_version_ge

from mindformers.models.glm.attention import RotaryEmbeddingFP32SoftmaxSelfAttention


class userskv(nn.Cell):
    def __init__(self, kv_shape, params_dtype):
        super(userskv, self).__init__()
        self.k_past = Parameter(Tensor(np.zeros(shape=kv_shape), params_dtype))
        self.value_past = Parameter(Tensor(np.zeros(shape=kv_shape), params_dtype))


class GEGLU(nn.Cell):
    """GEGLU activation"""

    def __init__(self, parallel_config):
        super(GEGLU, self).__init__()
        self.split = ops.Split(output_num=2, axis=-1)
        self.activation_fn = nn.GELU().gelu
        self.parallel_config = parallel_config

    def construct(self, x):
        x1, x2 = self.split(x)
        return x1 * self.activation_fn(x2)


class MLPWithGEGLU(nn.Cell):
    """MLP layer with GEGLU"""
    def __init__(self,
                 hidden_size,
                 output_dropout_prob,
                 inner_hidden_size=None,
                 layer_id=None,
                 bias=True,
                 activation_func='GELU',
                 params_dtype=mstype.float32,
                 compute_dtype=mstype.float16,
                 parallel_config=None):
        super(MLPWithGEGLU, self).__init__()
        self.layer_id = layer_id
        # Project to 4h.
        self.hidden_size = hidden_size

        if inner_hidden_size is None:
            inner_hidden_size = 4 * hidden_size

        self.inner_hidden_size = inner_hidden_size
        if activation_func == 'GEGLU':
            self.activation_func = GEGLU(parallel_config)
            h_to_4h_output_channel = 2 * self.inner_hidden_size
        elif activation_func == 'GELU':
            self.activation_func = nn.GELU()
            self.activation_func.gelu.shard(((parallel_config.data_parallel, 1, parallel_config.model_parallel),))
            h_to_4h_output_channel = self.inner_hidden_size

        self.dense_h_to_4h = Linear(
            self.hidden_size,
            h_to_4h_output_channel,
            has_bias=bias,
            transpose_b=True,
            param_init_type=params_dtype,
            compute_dtype=compute_dtype,
        )
        self.dense_h_to_4h.shard(
            strategy_matmul=((parallel_config.data_parallel, 1), (parallel_config.model_parallel, 1)),
            strategy_bias=((parallel_config.data_parallel, parallel_config.model_parallel),
                           (parallel_config.model_parallel,)))

        # Project back to h.
        self.dense_4h_to_h = Linear(
            self.inner_hidden_size,
            self.hidden_size,
            has_bias=bias,
            param_init_type=params_dtype,
            compute_dtype=compute_dtype,
        )
        self.dense_4h_to_h.shard(
            strategy_matmul=((parallel_config.data_parallel, parallel_config.model_parallel),
                             (1, parallel_config.model_parallel)),
            strategy_bias=((parallel_config.data_parallel, 1), (1,))
            )

        if is_version_ge(ms.__version__, '1.11.0'):
            self.dropout = nn.Dropout(p=output_dropout_prob)
        else:
            self.dropout = nn.Dropout(keep_prob=1 - output_dropout_prob)
        self.dropout.dropout.shard(((parallel_config.data_parallel, parallel_config.model_parallel),))

    def mlp_forward(self, hidden_states):
        """mlp forward."""
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        output = self.dense_4h_to_h(intermediate_parallel)
        return output

    def construct(self, hidden_states):
        output = self.mlp_forward(hidden_states)

        if self.training:
            output = self.dropout(output)
        return output


class DeepNormWithGLULayer(nn.Cell):
    """
    GLM base layer

    Args:
        num_layers (int): Number of layers.
        hidden_size (int): Hidden layer size.
        num_attention_heads (int): Number of attention heads.
        batch_size (int): Batch size.
        attention_dropout_prob (float, [0, 1.0]): Attention layer dropout probability.
        output_dropout_prob (float, [0, 1.0]): Output dropout probability.
        layernorm_epsilon (float): Layernorm epsilon.
        layer_id (int): Layer id.
        max_seq_len (int): Max sequence length.
        inner_hidden_size (optional): Inner hidden layer size. Default: None.
        hidden_size_per_attention_head (optional): Default: None.
        layernorm_order (str, optional): Which order to use layernorm. Default: 'pre'.
        layernorm (optional): Layernorm function. Default: LayerNorm.
        use_bias (bool, optional): Use bias or not. Default: True.
        activation_func (str, optional, 'GEGLU'/'GELU'): Choose activation function. Default: 'GEGLU'.
        position_encoding_2d (bool, optional): Use 2d position encoding or not. Default: True.
        params_dtype (ms.dtype, optional): Parameter data type. Default: mstype.float32.
        layernorm_dtype (ms.dtype, optional): Calculate layernorm data type. Default: mstype.float32.
        softmax_dtype (ms.dtype, optional): Calculate softmax data type. Default: mstype.float32.
        compute_dtype (ms.dtype, optional): Other compute data type. Default: mstype.float16.
        parallel_config (optional): Operator parallel strategy, Default: None.
        use_past (bool, optional): Use infer cache or not. Default: False.
    """

    def __init__(self,
                 num_layers,
                 hidden_size,
                 num_attention_heads,
                 batch_size,
                 attention_dropout_prob,
                 output_dropout_prob,
                 layernorm_epsilon,
                 layer_id,
                 max_seq_len=512,
                 inner_hidden_size=None,
                 hidden_size_per_attention_head=None,
                 layernorm_order='pre',
                 layernorm=LayerNorm,
                 use_bias=True,
                 activation_func='GEGLU',
                 position_encoding_2d=True,
                 params_dtype=mstype.float32,
                 layernorm_dtype=mstype.float32,
                 softmax_dtype=mstype.float32,
                 compute_dtype=mstype.float16,
                 parallel_config=None,
                 use_past=False,
                users=0):
        super(DeepNormWithGLULayer, self).__init__()
        # Set output layer initialization if not provided.
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.layer_id = layer_id
        if inner_hidden_size is None:
            inner_hidden_size = 4 * hidden_size * 2 // 3
        self.inner_hidden_size = inner_hidden_size
        self.position_encoding_2d = position_encoding_2d
        self.layernorm_order = layernorm_order
        self.use_past = use_past

        self.params_dtype = params_dtype
        self.layernorm_dtype = layernorm_dtype
        self.softmax_dtype = softmax_dtype
        self.compute_dtype = compute_dtype

        # Layernorm on the input data.
        self.input_layernorm = layernorm(hidden_size, param_init_type=layernorm_dtype, eps=layernorm_epsilon)
        self.input_layernorm.shard(((parallel_config.data_parallel, 1, 1),))

        # Self attention.
        self.attention = RotaryEmbeddingFP32SoftmaxSelfAttention(
            hidden_size,
            batch_size,
            num_attention_heads,
            parallel_config,
            attention_dropout_prob,
            output_dropout_prob,
            layer_id,
            hidden_size_per_attention_head=hidden_size_per_attention_head,
            position_encoding_2d=self.position_encoding_2d,
            bias=use_bias,
            params_dtype=params_dtype,
            softmax_dtype=softmax_dtype,
            compute_dtype=compute_dtype,
            max_seq_len=max_seq_len,
            use_past=use_past,
        )

        # Layernorm on the input data.
        self.post_attention_layernorm = layernorm(hidden_size, param_init_type=layernorm_dtype, eps=layernorm_epsilon)
        self.post_attention_layernorm.shard(((parallel_config.data_parallel, 1, 1),))
        if self.layernorm_order == 'sandwich':
            self.third_layernorm = layernorm(hidden_size, param_init_type=layernorm_dtype, eps=layernorm_epsilon)
            self.fourth_layernorm = layernorm(hidden_size, param_init_type=layernorm_dtype, eps=layernorm_epsilon)

        # MLP
        self.mlp = MLPWithGEGLU(
            hidden_size,
            output_dropout_prob,
            inner_hidden_size=inner_hidden_size,
            bias=use_bias,
            layer_id=layer_id,
            activation_func=activation_func,
            params_dtype=params_dtype,
            parallel_config=parallel_config
        )

        self.key_past = None
        self.value_past = None
        if self.use_past:
            # operator used for state reuse
            self.reducesum = P.ReduceSum().shard(((1, 1, 1, 1),))
            self.not_equal = P.NotEqual().shard(((1, 1, 1, 1), ()))
            self.slice = P.StridedSlice().shard(((1, 1, 1, 1),))
            size_per_head = int(hidden_size / num_attention_heads)
            self.kv_shape = (batch_size, num_attention_heads, size_per_head, max_seq_len)
            # parameters saving key and value states
            
            self.kv_past = nn.CellList()
            
            for tmp in range(users):
                self.kv_past.append(userskv(self.kv_shape, self.params_dtype))
                
                
            # self.key_past = Parameter(Tensor(np.zeros(shape=self.kv_shape), self.params_dtype), name="key_past")
            # self.value_past = Parameter(Tensor(np.zeros(shape=self.kv_shape), self.params_dtype), name="value_past")
            self.tile = P.Tile().shard(((1, 1),))
            self.mul = P.Mul().shard(((1, 1, 1, 1), ()))
            self.assign = P.Assign().shard(((1, 1, 1, 1), (1, 1, 1, 1)))
        self.mul = P.Mul()
        self.mul.shard(((parallel_config.data_parallel, 1, 1), ()))
        self.mul_4 = P.Mul()
        self.mul_4.shard(((parallel_config.data_parallel, 1, 1, 1), (parallel_config.data_parallel,)))

    def layer_forward(self, hidden_states, mask, position_ids, init_reset=True, batch_valid_length=None, user=0):
        """
            hidden_states: [seq_len, batch, hidden_size] with (1, dp, 1)
            mask: [(1, 1), seq_len, seq_len]
        Inputs:
            hidden_states (Tensor): Hidden layer output.
            mask (Tensor): Used when batching sequences together.
            position_ids (Tensor): Used to identify each token's position in the list of tokens.
            init_reset (bool, optional): Default: True.
            batch_valid_length (Tensor, optional): Default: None.

        Return:
            output (Tensor): Layer output.
        """
        # Layer norm at the beginning of the transformer layer.
        attention_input = self.input_layernorm(hidden_states)

        # indicate whether reset saved states
        key_reset = None
        value_reset = None

        if self.use_past:
            # reset states, init_reset True for reuse and False for reset
            key_reset = self.assign(self.kv_past[user].k_past, self.mul_4(self.kv_past[user].k_past, F.cast(init_reset, self.params_dtype)))
            value_reset = self.assign(self.kv_past[user].value_past,
                                      self.mul_4(self.kv_past[user].value_past, F.cast(init_reset, self.params_dtype)))
            # add dependency for desired execution order
            attention_input = F.depend(attention_input, key_reset)
            attention_input = F.depend(attention_input, value_reset)
        # t1 = self.key_past
        # tmp = self.key_past[user].value()
        # k_tmp = Parameter(self.key_past[user].value(), name="k_tmp")
        # v_tmp = Parameter(self.value_past[user].value(), name="v_tmp")
        attention_output, layer_present = self.attention(attention_input, mask, position_ids, self.layer_id,
                                                        self.kv_past[user].k_past, self.kv_past[user].value_past, batch_valid_length)

        # Residual connection.
        alpha = (2 * self.num_layers) ** 0.5
        hidden_states = self.mul(attention_input, alpha) + attention_output

        mlp_input = self.post_attention_layernorm(hidden_states)

        # MLP.
        mlp_output = self.mlp(mlp_input)

        value_update = None
        key_update = None
        if self.use_past:
            # current key and value
            key_present, value_present = layer_present
            # update key and value calculated this step
            key_update = self.assign(self.kv_past[user].k_past, key_present)
            value_update = self.assign(self.kv_past[user].value_past, value_present)
            # add dependency for desired execution order
            key_update = F.depend(key_update, key_reset)
            value_update = F.depend(value_update, value_reset)

        # add dependency for desired execution order
        mlp_output = F.depend(mlp_output, value_update)
        mlp_output = F.depend(mlp_output, key_update)
        # Second residual connection.
        output = mlp_input * alpha + mlp_output

        return output

    def construct(self, hidden_states, mask, position_ids, init_reset=True, batch_valid_length=None, user=0):
        '''
            hidden_states: [batch, seq_len, hidden_size]
            mask: [(1, 1), seq_len, seq_len]
        '''
        return self.layer_forward(hidden_states, mask, position_ids, init_reset, batch_valid_length, user)
```

## 3. 接口访问

```
pip install fastapi "uvicorn[standard]"
```

### 3.1 接口

```python
import ast
import threading
import time

import mindspore as ms
import numpy as np
import uvicorn
from fastapi import FastAPI
from mindformers.models.glm import GLMConfig, GLMChatModel
from mindformers.models.glm.chatglm_6b_tokenizer import ChatGLMTokenizer
from pydantic import BaseModel
from starlette.responses import StreamingResponse
import queue

tlock = threading.Lock()



app = FastAPI()


users = 2

avai_users = queue.Queue()
for i in range(users):
    avai_users.put(i)



class QueryInfo(BaseModel):
    query: str
    history: str


# 生成数据流
def generate_data(generation_kwargs):
    result = ''
    for new_text in model._stream_chat(**generation_kwargs):
        print(generation_kwargs['qi'],time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), new_text)
        yield new_text
    avai_users.put(generation_kwargs['qi'])
    
def generate_data_fun(generation_kwargs):
    result = ''
    for new_text in model._stream_chat(**generation_kwargs):
        print(generation_kwargs['qi'],time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), new_text)


@app.post("/stream")
def create_item(queryinfo: QueryInfo):
    query = queryinfo.query
    history = queryinfo.history
    
    try:
        user = avai_users.get(timeout=6)
    except queue.Empty:
        return Response("exit")

    print(history, type(history))
    if history == '[]':
        prompt = query
    else:
        history = ast.literal_eval(history)
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
        prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
    inputs = tokenizer(prompt)
    generation_kwargs = dict(
        origin_inputs=np.expand_dims(np.array(inputs['input_ids']).astype(np.int32), 0),
        top_k=1,
        top_p=1,
        repetition_penalty=1,
        max_length=config.seq_length,
        eos_token_id=config.eos_token_id,
        pad_token_id=config.pad_token_id,
        tokenizer=tokenizer, tlock=tlock, qi=user)
    sr = generate_data(generation_kwargs)

    return StreamingResponse(sr, media_type="text/plain")


if __name__ == '__main__':

    config = GLMConfig(
        position_encoding_2d=True,
        use_past=True,
        is_npu_acceleration=True,
        users=users
    )
    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=1)
    model = GLMChatModel(config)
    ms.load_checkpoint("/home/ma-user/work/MindFormers/checkpoint_download/glm/glm_6b.ckpt", model)
    tokenizer = ChatGLMTokenizer('/home/ma-user/work/checkpoint_download/glm/ice_text.model')
    inputs = tokenizer("你好")
    
    for i in range(users):
        generation_kwargs = dict(
            origin_inputs=np.expand_dims(np.array(inputs['input_ids']).astype(np.int32), 0),
            top_k=1,
            top_p=1,
            repetition_penalty=1,
            max_length=config.seq_length,
            eos_token_id=config.eos_token_id,
            pad_token_id=config.pad_token_id,
            tokenizer=tokenizer, tlock=tlock, qi=i)
        th = threading.Thread(target=generate_data_fun, args=(generation_kwargs,))
        th.start()

    # for i in model._stream_chat(
    #         origin_inputs=np.expand_dims(np.array(inputs['input_ids']).astype(np.int32), 0),
    #         top_k=1,
    #         top_p=1,
    #         repetition_penalty=1,
    #         max_length=config.seq_length,
    #         eos_token_id=config.eos_token_id,
    #         pad_token_id=config.pad_token_id,
    #         tokenizer=tokenizer, tlock=tlock):
    #     print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), i)
    uvicorn.run(app, host='0.0.0.0', port=30000, workers=1)
```

### 3.2 测试接口

```python
import threading
import time

import requests

import requests
from threading import Thread

url = 'http://127.0.0.1:30000/stream/'



def f(n):
    query = {0:"什么叫好天气",
             1:'柯南是谁',
             2:'天空是什么颜色的',
             3:'浙江在哪',
             4:'你好',
             5:'静夜思怎么背',
             6:'春节是什么时候',
             }[n]
    query = "请问答问题，字数不要超过60字，问题："+query
    data = {
        "query": query,
        "history": '[]'
    }
    rsp = requests.post(url, json=data, stream=True)
    start_time = time.time()
    count = 0
    for chunk in rsp.iter_content(chunk_size=2000):
        if chunk:
            print(str(n)*10,chunk.decode('utf-8'))  # 在这里对接收到的数据进行处理，例如打印出来
            count+=1
    end_time = time.time()
    print(f'generate speed: {count/(end_time-start_time):.2f} tokens/s')
    print(f'{n}'*10,"over")

for i in range(7):
    th = threading.Thread(target=f, args=(i,))
    th.start()
```

![](https://cleansely.top:48083/i/2023/08/17/64ddcc8110fb6.jpg)

