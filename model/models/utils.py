import os
import random
import numpy as np
import torch
import torch.distributed as dist

def fix_random_seed(seed: int):
    """Fix random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def setup_distributed():
    """Setup distributed training environment."""
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    return rank

def _to_int_tuple(s: str):
    return tuple(map(int, s.split(",")))