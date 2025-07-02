import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import os

from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from dataset import FolderDataset
from model import EncoderForPretraining
from tqdm import tqdm

from dataclasses import dataclass

@dataclass
class Config:
    # Training Configurations
    data_path: str = '../datasets'
    batch_size: int = 128
    max_step: int = 10000
    accumulate_batch_num: int = 1
    lr: float = 1e-3
    min_lr: float = 1e-6
    weight_decay: float = 0.0
    logger: str = 'wandb'
    output_dir: str = 'output'
    
    # Model Configurations
    encoder_cfg_path: str = '/cpfs02/shared/speechllm/liuzhan/workspace_sci/icefall_general_encoder/egs/general_audio_encoder/mtl/zipformer_audio_encoder/whisper-encoder/whisper-encoder-146M'
    encoder_ckpt_path: str = None
    feature_dim: int = 1024
    num_classes: int = 4

def setup_ddp():
    if 'LOCAL_RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        return local_rank
    else:
        return None

def is_distributed():
    """检查是否为分布式训练"""
    return dist.is_available() and dist.is_initialized()

class LoggerWrapper:
    def __init__(self, logger):
        self.logger = logger

    def log(self, info: dict):
        self.logger.info(info)

def init_logger(config):
    if config.logger == 'wandb':
        import wandb
        wandb.init(project="encoder-contrast")
        logger = wandb
    else:
        import logging
        logger = logging.getLogger('EncoderContrast')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler('encoder_contrast.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.handlers = []
        logger.addHandler(handler)
        logger = LoggerWrapper(logger)
    return logger

def build_dataloader(config, is_ddp=False, rank=0):
    ds = FolderDataset(config.data_path, transform=None)
    
    if is_ddp:
        # 分布式训练使用DistributedSampler
        sampler = DistributedSampler(ds, shuffle=True)
    else:
        # 单卡训练使用WeightedRandomSampler
        labels = [entry[1] for entry in ds.samples]
        class_sample_count = np.bincount(labels)
        weights = 1. / class_sample_count[labels]
        sampler = WeightedRandomSampler(weights, len(weights))

    def collate_fn(batch):
        waveforms, labels, sub_labels = zip(*[(entry['inputs'], entry['class_idx'], entry['subclass_idx']) for entry in batch])
        lengths = [w.shape[0] for w in waveforms]
        max_len = max(lengths)
        padded = []
        pad_mask = torch.zeros(len(waveforms), max_len)
        
        for i, w in enumerate(waveforms):
            pad = torch.zeros(max_len)
            pad[:w.shape[0]] = w
            padded.append(pad)
            pad_mask[i, w.shape[0]:] = 1

        return {
            'inputs': torch.stack(padded),
            'class_idx': torch.tensor(labels),
            'subclass_idx': torch.tensor(sub_labels),
            'pad_mask': pad_mask
        }

    dl = DataLoader(ds, batch_size=config.batch_size, collate_fn=collate_fn, sampler=sampler)
    return dl

def create_optimizer_scheduler(model, config):
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_step, eta_min=config.min_lr)
    return optimizer, scheduler

def main():
    # 尝试初始化分布式训练
    local_rank = setup_ddp()
    is_ddp = local_rank is not None
    
    if is_ddp:
        world_size = dist.get_world_size()
        device = torch.device(f'cuda:{local_rank}')
        print(f"Using DDP with {world_size} GPUs, local rank: {local_rank}")
    else:
        local_rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using single GPU/CPU: {device}")
    
    config = Config()

    # 只在主进程或单卡时初始化logger
    if local_rank == 0:
        logger = init_logger(config)
    else:
        logger = None

    dataloader = build_dataloader(config, is_ddp=is_ddp, rank=local_rank)
    model = EncoderForPretraining(config)
    model.to(device)
    
    # 根据是否分布式训练包装模型
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    model.train()
    optimizer, scheduler = create_optimizer_scheduler(model, config)

    max_step = config.max_step
    step_count = 0

    while True:
        # 只有在分布式训练时才设置epoch
        if is_ddp and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(step_count // len(dataloader))

        outputs_cache = []
        labels_cache = []

        # 只在主进程显示进度条
        pbar = tqdm(dataloader, desc="Training", disable=(is_ddp and local_rank != 0))
        
        for i, batch in enumerate(pbar):
            if step_count >= max_step:
                break
            
            inputs = batch['inputs'].to(device)
            class_idx = batch['class_idx'].to(device)
            subclass_idx = batch['subclass_idx'].to(device)
            pad_mask = batch['pad_mask'].to(device)
            
            # 根据是否使用DDP调用模型
            if is_ddp:
                embeds = model.module.encode(inputs, pad_mask)
            else:
                embeds = model.encode(inputs, pad_mask)
            
            outputs_cache.append(embeds)
            labels_cache.append(class_idx)

            if len(outputs_cache) < config.accumulate_batch_num:
                continue
            
            outputs = torch.cat(outputs_cache, dim=0)
            labels = torch.cat(labels_cache, dim=0)

            # 根据是否使用DDP计算损失
            if is_ddp:
                con_loss = model.module.cal_loss(outputs, labels, loss_type='supcon')  
                cla_loss = model.module.cal_loss(outputs, labels, loss_type='classify')
            else:
                con_loss = model.cal_loss(outputs, labels, loss_type='supcon')
                cla_loss = model.cal_loss(outputs, labels, loss_type='classify')
            
            loss = con_loss + cla_loss
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            outputs_cache = []
            labels_cache = []

            # 只在主进程记录日志
            if local_rank == 0 and logger is not None:
                logger.log({
                    "train/loss": loss.item(), 
                    "train/con_loss": con_loss.item(),
                    "train/cla_loss": cla_loss.item(),
                    "step": step_count, 
                    "train/lr": optimizer.param_groups[0]['lr']
                })
            
            step_count += 1
        
        if step_count >= max_step:
            break 
    
    # 只在主进程保存模型
    if local_rank == 0:
        if not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir)
        save_path = os.path.join(config.output_dir, 'model_final.pth')
        
        # 保存模型时注意DDP包装
        if is_ddp:
            torch.save(model.module.state_dict(), save_path)
        else:
            torch.save(model.state_dict(), save_path)

    # 只有分布式训练时才需要销毁进程组
    if is_ddp:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()