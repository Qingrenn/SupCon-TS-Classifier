from transformers import WhisperConfig
from .models.whisper_model import CustomWhisperEncoder
from .models.utils import _to_int_tuple
from .head import SupConHead, ClassificationHead, FocalSupConHead, FocalLossHead, HierarchicalSupConHead

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T


# Downsample
class DwonsampleModule(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.Sequential(
            # 第一层卷积，增加通道数并开始降采样
            nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            # 第二层卷积，进一步降采样
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(64),
            # 第三层卷积，继续降采样
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(128),
            # 第四层卷积，达到最终降采样比例
            nn.Conv1d(in_channels=128, out_channels=out_channels, kernel_size=5, stride=5, padding=2),
            nn.BatchNorm1d(128)
        )
    
    def forward(self, x):
        # x: (batch, time, channel)
        x = x.permute(0, 2, 1)  # (batch, channel, time)
        x = self.downsample(x)  # (batch, channel, time // 160)
        x = x.permute(0, 2, 1)  # (batch, time // 160, channel)
        return x


# Backbone
def get_encoder_model(whisper_encoder_cfg_path, encoder_dim="192,384,768,1024,768,384", pretrained_ckpt=None) -> nn.Module:
    config = WhisperConfig.from_pretrained(whisper_encoder_cfg_path)
    encoder = CustomWhisperEncoder(config)
    if pretrained_ckpt is not None:
        pass
    encoder.add_adpt_out(max(_to_int_tuple(encoder_dim)))
    encoder.add_adpt_in(128)
    return encoder


class EncoderForPretraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.downsample = DwonsampleModule(in_channels=1, out_channels=128)  # Assuming input is mono audio
        self.encoder = get_encoder_model(config.encoder_cfg_path, pretrained_ckpt=config.encoder_ckpt_path)
        self.supcon_head = HierarchicalSupConHead(temperature=0.07, base_temperature=0.07)
        self.class_head = FocalLossHead(hidden_states=config.feature_dim, num_classes=config.num_classes)

        self.neck = nn.Sequential(
            nn.Linear(config.feature_dim, 2*config.feature_dim, bias=False),
            nn.ReLU(),
            nn.Linear(2*config.feature_dim, config.feature_dim, bias=False)
        )

    def encoder_forward(self, x):
        '''
            x: (batch, time)
        '''
        x = x.unsqueeze(-1)  # (batch, time, channel)
        x = self.downsample(x) # Downsample to (batch, time // 160, channel)
        encoder_out, encoder_out_lens, _ = self.encoder(
            x, output_hidden_states=True
        )
        encoder_out = encoder_out.permute(1, 0, 2) # (batch, time, dim)
        return encoder_out

    def __aggregate_embeddings__(self, embeddings, pad_mask):
        '''
            embeddings: (batch, time, dim)
            pad_mask: (batch, time) 1 for padded, 0 for unpadded

            returns: (batch, dim)
        '''
        if pad_mask is None:
            return embeddings.mean(dim=1)  # (batch, dim)

        downsampled_mask = F.adaptive_avg_pool1d(pad_mask.float(), embeddings.shape[1])
        downsampled_mask = downsampled_mask > 0
        unpadded_mask = ~ downsampled_mask
        batch_list = [xi[mi] for xi, mi in zip(embeddings, unpadded_mask)]
        embeddings = torch.concat([entry.mean(dim=0, keepdim=True) for entry in batch_list], dim=0)
        
        return embeddings

    def encode(self, x, pad_mask=None):
        embeddings = self.encoder_forward(x)
        embeddings = self.__aggregate_embeddings__(embeddings, pad_mask)
        embeddings = self.neck(embeddings)  # (batch, dim)
        embeddings = F.normalize(embeddings, dim=-1, p=2) # L2 Normalize embeddings
        return embeddings

    def cal_loss(self, embeddings, labels, sublabels, loss_type='supcon'):
        '''
            embeddings: (batch, dim)
            labels: (batch,)
        '''
        assert loss_type in ['supcon', 'classify'], f"Unknown loss type: {loss_type}"
        
        if loss_type == 'classify':
            loss = self.class_head(embeddings, labels)['loss']
            return loss

        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(1)
        
        loss = self.supcon_head(embeddings, labels, sublabels)['loss']
        return loss

    def forward(self, x, labels=None, pad_mask=None):
        # x: (batch, time)
        # labels: (batch,)
        # sub_labels: (batch,)
        # pad_mask: (batch, time)

        embeddings = self.encode(x, pad_mask)
        logits = self.class_head(embeddings)['logits']

        if labels is not None:
            loss = self.cal_loss(embeddings, labels, loss_type='supcon')
        else:
            loss = None

        return {
            'embeddings': embeddings,
            'loss': loss,
            'logits': logits
        }
    
    def cal_loss_with_gathered_features(self, 
        local_features, gathered_features, gathered_labels, gathered_sublabels, 
        start_idx, end_idx, loss_type='supcon'):
        """
        使用聚合后的特征计算损失，但只对本地样本计算梯度
        
        Args:
            gathered_features: 聚合后的特征 [global_batch_size, feature_dim]
            gathered_labels: 聚合后的标签 [global_batch_size]
            gathered_sublabels: 聚合后的子标签 [global_batch_size]
            start_idx: 本地样本在全局batch中的起始索引
            end_idx: 本地样本在全局batch中的结束索引
            loss_type: 损失类型
        """
        
        return self.supcon_head.forward_with_gathered_features(
            local_features, gathered_features, gathered_labels, gathered_sublabels, start_idx, end_idx,
        )['loss']


class EncoderForClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = EncoderForPretraining(config)

        self.class_heads = nn.ModuleList()
        for k, v in config.task2class.items():
            class_head = FocalLossHead(
                hidden_states=config.feature_dim, 
                num_classes=v,
            )
            self.class_heads.append(class_head)
        
        self.loss_weights = config.loss_weights if hasattr(config, 'loss_weights') else None

    def forward(self, x, task_id, labels=None, pad_mask=None):
        # x: (batch, time)
        # task: (batch,)
        # labels: (batch,)
        # pad_mask: (batch, time)

        embeddings = self.encoder.encode(x, pad_mask)
        
        unique_task_ids = list(set(task_id.cpu().numpy()))
        task2output = {}

        for id in unique_task_ids:
            task_mask = (task_id == id)
            if labels is not None:
                task_labels = labels[task_mask]
            else:
                task_labels = None

            output = self.class_heads[id](x=embeddings[task_mask], labels=task_labels)
            
            task2output[id] = {
                'embeddings': embeddings[task_mask],
                'logits': output['logits'],
                'loss': output['loss'],
            }

        
        total_loss = []
        for id, output in task2output.items():
            weight = self.loss_weights[id] if self.loss_weights else 1.0
            if output['loss'] is not None:
                total_loss.append(weight * output['loss'])
        total_loss = torch.stack(total_loss).mean() if total_loss else None 

        return {
            'outputs': task2output,
            'total_loss': total_loss
        }