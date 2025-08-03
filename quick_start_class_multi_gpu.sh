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

export NCCL_TIMEOUT=600
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1


cd /cpfs04/user/yaoqingren/workspace/encoder-contrast/encoder_contrast_train/
echo "[√] 已切换工作目录到 `pwd`"

PYTHON="/root/miniconda3/envs/icefall/bin/python"
echo "[√] 使用 Python 路径: $PYTHON"

"$PYTHON" -m torch.distributed.run --nproc_per_node=8 train_class.py --resume local_checkpoint/output_foccon_only_exp9/model_final.pth