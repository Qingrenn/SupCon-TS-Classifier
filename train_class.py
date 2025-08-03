import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import os
import argparse

from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.nn.functional import all_gather

from data import FolderDataset
from model import EncoderForClassification
from tqdm import tqdm

from dataclasses import dataclass, field

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, default='local_checkpoint/class_exp2')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--max_step', type=int, default=10000)
    parser.add_argument('--logger', type=str, default='wandb')
    parser.add_argument('--resume', type=str, default=None, help='Path to resume training from a checkpoint')
    return parser.parse_args()

@dataclass
class Config:
    # Training Configurations
    train_data_path: str = 'datasets/train_datasets'
    val_data_path: str = 'datasets/test_subdatasets'
    batch_size: int = 128
    max_step: int = 10000
    accumulate_batch_num: int = 1
    lr: float = 3e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.0
    logger: str = 'wandb'
    output_dir: str = 'local_checkpoint/output'
    resume: str = None  # Path to resume training from a checkpoint
    
    # Model Configurations
    encoder_cfg_path: str = '/cpfs02/shared/speechllm/liuzhan/workspace_sci/icefall_general_encoder/egs/general_audio_encoder/mtl/zipformer_audio_encoder/whisper-encoder/whisper-encoder-146M'
    encoder_ckpt_path: str = '/cpfs02/shared/speechllm/liuzhan/workspace_sci/icefall_general_encoder/egs/general_audio_encoder/mtl/zipformer_audio_encoder/exp-ds-xlarge-lr-0.02-full-en-zh-audio-multi-kd-time-mask-ratio-2.0-shar/iter-352000-avg-2.pt'
    feature_dim: int = 1024
    num_classes: int = 4
    task2class: dict = field(default_factory=lambda: {
        'gw': 2,
        'leaves': 7,
        'sleep': 5,
        'stead': 2
    })

    def update(self, updates_dict):
        """更新配置参数"""
        for key, value in updates_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Config has no attribute '{key}', ignoring...")


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

def build_dataloader(config, is_ddp=False, rank=0, mode='train'):
    if mode == 'train':
        ds = FolderDataset(config.train_data_path, transform=None)
    elif mode == 'val':
        ds = FolderDataset(config.val_data_path, transform=None)
    else:
        raise ValueError("Unsupported mode: {}".format(mode))
    
    print('Class index:', ds.class_to_idx)
    print('Sub-class index:', ds.subclass_to_idx)
    
    if mode == 'train':
        # 计算类别权重 - 每个类别被采样到的概率相同
        labels = [entry[1]*10 + entry[2] for entry in ds.samples]
        class_sample_count = np.bincount(labels)
        class_weights = len(class_sample_count) / class_sample_count
        sample_weights = class_weights[labels]
        
        if is_ddp:
            # 分布式训练：先分配数据到各个rank，然后在每个rank内进行类别平衡采样
            rank_indices = list(range(rank, len(ds), dist.get_world_size()))
            rank_weights = [sample_weights[i] for i in rank_indices]
            rank_dataset = Subset(ds, rank_indices)
            sampler = WeightedRandomSampler(rank_weights, len(rank_weights), replacement=True)
            ds = rank_dataset  # 更新数据集为当前rank的子集
        else:
            # 单卡训练：直接使用类别平衡采样
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    else:
        # 验证集不需要类别平衡采样，直接使用顺序采样
        sampler = None

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

    dl = DataLoader(
        ds, 
        batch_size=config.batch_size, 
        collate_fn=collate_fn, 
        sampler=sampler, 
        drop_last=True
    )
    
    return dl

def create_optimizer_scheduler(model, config):
    if is_distributed():
        # optimizer = optim.AdamW(model.module.class_heads.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        optimizer = optim.AdamW(model.module.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        # optimizer = optim.AdamW(model.class_heads.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # 预热调度器：100步内从0线性增长到目标学习率
    warmup_step = int(config.max_step * 0.01)
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.01,  # 从1%的学习率开始
        end_factor=1.0,     # 到100%的学习率
        total_iters=warmup_step    # 100步预热
    )

    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.max_step - warmup_step,  # 剩余步数
        eta_min=config.min_lr
    )

    # 组合调度器
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[100]  # 在第100步切换
    )

    return optimizer, scheduler


def preprocess_ckpt(ckpt):
    new_ckpt = {}
    for k, v in ckpt.items():
        if k.startswith('class_head'):
            continue
        new_ckpt['encoder.' + k] = v
    return new_ckpt


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
    args = parse_args()
    config.update(vars(args))

    # 只在主进程或单卡时初始化logger
    if local_rank == 0:
        logger = init_logger(config)
    else:
        logger = None

    dataloader = build_dataloader(config, is_ddp=is_ddp, rank=local_rank)
    val_dl = build_dataloader(config, is_ddp=is_ddp, rank=local_rank, mode='val')
    model = EncoderForClassification(config)
    if config.resume:
        if os.path.isfile(config.resume):
            ckpt = torch.load(config.resume)
            ckpt = preprocess_ckpt(ckpt)
            outputs = model.load_state_dict(ckpt, strict=False)
            print(f"Resuming from checkpoint: {config.resume}, missing keys: {outputs.missing_keys}")
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
        sublabels_cache = []

        # 只在主进程显示进度条
        pbar = tqdm(dataloader, desc="Training", disable=(is_ddp and local_rank != 0))
        
        for i, batch in enumerate(pbar):
            if step_count >= max_step:
                break
            
            inputs = batch['inputs'].to(device)
            class_idx = batch['class_idx'].to(device)
            subclass_idx = batch['subclass_idx'].to(device)
            pad_mask = batch['pad_mask'].to(device)

            # 根据是否使用DDP计算损失
            if is_ddp:
                output = model.module(inputs, task_id=class_idx, labels=subclass_idx, pad_mask=pad_mask)
            else:
                output = model(inputs,  task_id=class_idx, labels=subclass_idx, pad_mask=pad_mask)

            loss = output['total_loss']
            task2loss = {f'train/{id}_loss': task_output['loss'].item() for id, task_output in output['outputs'].items()}

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # 只在主进程记录日志
            if local_rank == 0 and logger is not None:
                logger.log({
                    "train/loss": loss.item(), 
                    **task2loss,
                    "step": step_count, 
                    "train/lr": optimizer.param_groups[0]['lr']
                })
            
            step_count += 1
            EvalWhileTraining = False
            if EvalWhileTraining and local_rank ==0 and step_count % 100 == 0:
                # evaluate on validation set
                model.eval()
                val_outputs = []
                val_labels = []
                with torch.no_grad():
                    for val_batch in tqdm(val_dl, desc="Validating"):
                        val_inputs = val_batch['inputs'].to(device)
                        val_class_idx = val_batch['class_idx'].to(device)
                        val_pad_mask = val_batch['pad_mask'].to(device)

                        if is_ddp:
                            output = model.module(val_inputs, pad_mask=val_pad_mask)
                        else:
                            output = model(val_inputs, pad_mask=val_pad_mask)

                        val_outputs.append(torch.argmax(output['logits'], dim=-1))
                        val_labels.append(val_class_idx)
                
                val_outputs = torch.cat(val_outputs, dim=0)
                val_labels = torch.cat(val_labels, dim=0)
                oa = (val_outputs == val_labels).float().mean().item()
                unique_classes = torch.unique(val_labels)
                class_oa = {}
                for cl in unique_classes:
                    cls_mask = (val_labels == cl)
                    cls_oa = (val_outputs[cls_mask] == cl).float().mean().item()
                    class_oa[int(cl.item())] = cls_oa
                
                logger.log({
                    "val/overall_accuracy": oa,
                    **{f"val/class_{int(cl)}_accuracy": acc for cl, acc in class_oa.items()},
                    "step": step_count
                })

            if local_rank ==0 and step_count % 1000 == 0:
                if not os.path.exists(config.output_dir):
                    os.makedirs(config.output_dir)
                save_path = os.path.join(config.output_dir, f'model_{step_count}.pth')

                if is_ddp:
                    torch.save(model.module.state_dict(), save_path)
                else:
                    torch.save(model.state_dict(), save_path)


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