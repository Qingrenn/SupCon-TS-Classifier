#!/bin/bash

export LOCAL_RANK=0

set_proxy () {
    export https_proxy="https://fanke1:brVwNbRJq0Pj3WIVaPpdPrXd5LM4qnRjogwsBOXA3liJE9d4imxOZXeFtw8l@aliyun-proxy.pjlab.org.cn:13128"
    export http_proxy="https://fanke1:brVwNbRJq0Pj3WIVaPpdPrXd5LM4qnRjogwsBOXA3liJE9d4imxOZXeFtw8l@aliyun-proxy.pjlab.org.cn:13128"
    export HTTP_PROXY="https://fanke1:brVwNbRJq0Pj3WIVaPpdPrXd5LM4qnRjogwsBOXA3liJE9d4imxOZXeFtw8l@aliyun-proxy.pjlab.org.cn:13128"
    export HTTPS_PROXY="https://fanke1:brVwNbRJq0Pj3WIVaPpdPrXd5LM4qnRjogwsBOXA3liJE9d4imxOZXeFtw8l@aliyun-proxy.pjlab.org.cn:13128"
    echo "[√] 已设置 PJLAB HTTPS 代理"
}
set_proxy

cd /cpfs04/user/yaoqingren/workspace/encoder-contrast/encoder_contrast_train/
echo "[√] 已切换工作目录到 `pwd`"

PYTHON="/root/miniconda3/envs/icefall/bin/python"
echo "[√] 使用 Python 路径: $PYTHON"

"$PYTHON" -m torch.distributed.run --nproc_per_node=8 train.py