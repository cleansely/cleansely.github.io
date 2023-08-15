---
title: bms部署chatglm-ms-demo
date: 2023-08-15 18:04:00
tags: 
    - modelarts 
categories: 昇腾
---

# bms部署chatglm-ms-demo

## 1. ssh连接

ip：
user:root
pwd:

## 2. 换源

### 2.1 删除源文件

```
rm -f /etc/yum.repos.d/EulerOS-ISO.repo
```

### 2.2 新建源文件

`vi /etc/yum.repos.d/EulerOS.repo`

内容为
```
[base]
name=EulerOS-2.0SP8 base
baseurl=https://mirrors.huaweicloud.com/euler/2.8/os/aarch64/
enabled=1
gpgcheck=1
sslverify=0
gpgkey=https://mirrors.huaweicloud.com/euler/2.8/os/RPM-GPG-KEY-EulerOS
```

重建缓存
```
yum clean all
yum makecache
```

## 3. 更新固件驱动

### 3. 下载驱动

```
# 新建下载文件夹
cd /home
mkdir -p work/downloads
cd work/downloads/

# 下载
wget https://nbfae.obs.cn-east-xxx.nbaicc.com/pkgs/firmware_c85/Ascend-hdk-910-npu-driver_23.0.rc1_linux-aarch64.run
wget https://nbfae.obs.cn-east-xxx.nbaicc.com/pkgs/firmware_c85/Ascend-hdk-910-npu-firmware_6.3.0.1.241.run
wget https://nbfae.obs.cn-east-xxx.nbaicc.com/pkgs/Ascend-cann-toolkit_6.3.RC1_linux-aarch64.run
```

### 3.2 安装驱动

可执行权限
```
chmod +x Ascend-hdk-910-npu-firmware_6.3.0.1.241.run
chmod +x Ascend-hdk-910-npu-driver_23.0.rc1_linux-aarch64.run
chmod +x Ascend-cann-toolkit_6.3.RC1_linux-aarch64.run
```

升级安装
```
./Ascend-hdk-910-npu-firmware_6.3.0.1.241.run --upgrade
```

```
./Ascend-hdk-910-npu-driver_23.0.rc1_linux-aarch64.run --upgrade
```

```
./Ascend-cann-toolkit_6.3.RC1_linux-aarch64.run --install
```

### 3.3 重启服务器

`reboot`

## 4. 安装docker

### 4.1 yum安装docker

```
yum install -y docker
```

### 4.2 安装docker runtime

下载
``` 
cd /home/work/downloads
wget https://nbfae.obs.cn-east-xxx.nbaicc.com/pkgs/Ascend-docker-runtime_5.0.RC1_linux-aarch64.run
chmod +x Ascend-docker-runtime_5.0.RC1_linux-aarch64.run
```

安装
```
./Ascend-docker-runtime_5.0.RC1_linux-aarch64.run --install
```

### 4.3 重启docker

`service docker restart`


## 5. 安装obs工具

临时写入系统环境
```
mkdir -p /home/work/sofaware
cd /home/work/sofaware
wget https://obs-community.obs.cn-north-1.myhuaweicloud.com/obsutil/current/obsutil_linux_arm64.tar.gz

tar -zxvf obsutil_linux_arm64.tar.gz

cd obsutil_linux_arm64_5.4.11/

chmod 755 obsutil

export PATH=$PATH:/home/work/sofaware/obsutil_linux_arm64_5.4.11
```

## 6. 拉取chatglm权重

ak、sk要替换上，权重在自己的obs上可以下载

```
export PATH=$PATH:/home/work/sofaware/obsutil_linux_arm64_5.4.11

obsutil config -i= -k= -e=obs.cn-east-xxx.nbaicc.com

mkdir -p /home/work/weights

obsutil sync obs://lfx/weights/glm/ /home/work/weights/glm/

chmod 777 -R /home/work/weights/glm/
```

## 7. 拉取mindspore代码

### 7.1 安装git

`yum install -y git`

### 7.2 拉取代码

```
cd /home/work
git clone https://gitee.com/mindspore/mindformers.git
chmod 777 -R /home/work/mindformers
```

### 8. 拉取镜像

```
docker login -u cn-east-xxx@xxx -p xxx swr.cn-east-xxx.nbaicc.com

docker pull swr.cn-east-xxx.nbaicc.com/nbaicc_pub/ms20rc1_py39_cann63:2023062601
```

## 9. 新建docker容器

```
docker run -it \
    --device=/dev/davinci0 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/common \
    -v /usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/driver \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /home/work/weights/glm:/home/ma-user/work/checkpoint_download/glm \
    -v /home/work/mindformers:/home/ma-user/work/mindformers \
    -p 80:80 \
    --privileged \
    --entrypoint=/bin/bash \
    --user root \
swr.cn-east-xxx.nbaicc.com/nbaicc_pub/ms20rc1_py39_cann63:2023062601
```

## 10. 容器内操作

docker exec -it xxx /bin/bash

### 10.1 切换环境

```
export PATH="/home/ma-user/anaconda3/bin:$PATH"
source activate mindspore_py39
```

### 10.2 安装依赖

```
pip install fastapi "uvicorn[standard]"
```

### 10.3 新建web_demo.py

```
cd /home/ma-user/work
vi web_demo.py
```

内容如下

```
from fastapi import FastAPI, Request,Body
import uvicorn, json, datetime
import time
import mindspore as ms
import numpy as np
from mindformers.models.glm import GLMConfig, GLMChatModel
from mindformers.models.glm.chatglm_6b_tokenizer import ChatGLMTokenizer
from mindformers.models.glm.glm_processor import process_response
from pydantic import BaseModel, validator, conint, constr
import ast


class QueryInfo(BaseModel):
    query: str
    history: str


app = FastAPI(docs_url="/xx/docs",)


@app.post("/xx")
async def create_item(queryinfo:QueryInfo):
    query = queryinfo.query
    history = queryinfo.history

    print(history,type(history))
    if history == '[]':
        prompt = query
    else:
        history = ast.literal_eval(history)
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
        prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
    inputs = tokenizer(prompt)

    start_time = time.time()
    print(start_time)
    outputs = model.generate(np.expand_dims(np.array(inputs['input_ids']).astype(np.int32), 0),
                             max_length=config.max_decode_length, do_sample=False, top_p=0.7, top_k=1)
    end_time = time.time()
    print(f'不准，generate speed: {outputs[0].shape[0] / (end_time - start_time):.2f} tokens/s')
    response = tokenizer.decode(outputs)
    response = process_response(response[0])
    if history != '[]':
        history = history + [(query, response)]
    else:
        history = [(query, response),]
    print(response)

    answer = {
        "response": response,
        "history": str(history),
        "status": 200,
        # "time": time
    }
    return answer


if __name__ == '__main__':
    config = GLMConfig(
        position_encoding_2d=True,
        use_past=True,
        is_npu_acceleration=True,
    )
    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=0)
    model = GLMChatModel(config)
    ms.load_checkpoint("./checkpoint_download/ms_weight/ms_glm_6b.ckpt", model)
    tokenizer = ChatGLMTokenizer('./checkpoint_download/ms_weight/ice_text.model')
    # 初始化提问，节约时间
    
    # 初始化提问，节约时间
    uvicorn.run(app, host='0.0.0.0', port=30000, workers=1)
```

### 10.4 启动服务

启动需要些时间

``` 
cd /home/ma-user/work
# python web_demo.py
nohup python web_demo.py > api.log 2>&1 &
```

如图所示，表示启动成功
![](https://cleansely.top:48083/i/2023/08/15/64db4d09bc224.jpg)    


## 11. 访问接口方式

第一次回答较慢，history较长会报错

### 11.1 接口访问

自己电脑新建`api_test.py`

内容如下
```
import json

import requests
import json
import pprint
url = 'http://x.x.x.x/'

history = "[]"
while True:
    query = input("请提问,stop退出:")
    if query == 'stop':
        break
    data = {
        "query": query,
        "history": history
    }
    rsp = requests.post(url, json=data)

    pprint.pprint(json.loads(rsp.text)['response'])
    is_history = input("是否继续历史，0或者1：")
    if is_history == '0':
        history = '[]'
    else:
        history = str(json.loads(rsp.text)['history'])
        print(history)
```

![](https://cleansely.top:48083/i/2023/08/15/64db4cfd7bb5e.jpg)


### 11.2 fastapi自带网页访问

http://x.x.x.x/docs

body举例

```
{
  "query": "天空是什么颜色",
  "history": "[]"
}
```


![](https://cleansely.top:48083/i/2023/08/15/64db4cf628b0e.jpg)

![](media/16886261428403/16886319995700.jpg)


## 12. 后台日志输出

![image-20230815180108049](https://cleansely.top:48083/i/2023/08/15/64db4ce6114f5.png)
