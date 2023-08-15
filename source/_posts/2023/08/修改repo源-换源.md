---
title: 修改repo源 换源
date: 2023-08-15 09:06:38
tags: modelarts
categories: 昇腾
---

# 修改repo源

## 修改源

sudo vi /etc/yum.repos.d/EulerOS.repo

```
[base]
name=EulerOS-2.0SP8 base
baseurl=http://repo.huaweicloud.com/euler/2.8/os/aarch64/
enabled=1
gpgcheck=1
gpgkey=http://repo.huaweicloud.com/euler/2.8/os/RPM-GPG-KEY-EulerOS
```
sudo yum clean all
sudo yum makecache