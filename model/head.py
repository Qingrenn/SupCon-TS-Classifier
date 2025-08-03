import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn import preprocessing

"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""


class SupConHead(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConHead, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return {
            'loss': loss,
        }


class ClassificationHead(nn.Module):
    def __init__(self, hidden_states, num_classes):
        super().__init__()
        self.proj = nn.Linear(hidden_states, num_classes)

    def forward(self, x, labels=None):
        # x: (batch, feature_dim)
        logits = F.softmax(self.proj(x), dim=-1)
        
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        else:
            loss = None
        
        return {
            'logits': logits,
            'loss': loss
        }

class FocalSupConHead(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07, mixup=True):
        super(FocalSupConHead, self).__init__()
        self.temperature = temperature
        self.mixup = mixup
        self.base_temperature = base_temperature


    def forward(self, features, labels=None, mask=None):
        """
        Partial codes are based on the implementation of supervised contrastive loss. 
        import from https https://github.com/HobbitLong/SupContrast.
        """
        features = features.squeeze(1)  # (batch, feature_dim)
        device = (torch.device('cuda')
                if features.is_cuda
                else torch.device('cpu'))
        temperature=0.07
        base_temperature=0.07
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            if self.mixup:
                if labels.size(1)>1:
                    weight_index = 10**np.arange(args.num_classes)  
                    weight_index = torch.tensor(weight_index).unsqueeze(1).to("cuda")
                    labels_ = labels.mm(weight_index.float()).squeeze(1)
                    labels_ = labels_.detach().cpu().numpy()
                    le = preprocessing.LabelEncoder()
                    le.fit(labels_)
                    labels = le.transform(labels_)
                    labels=torch.unsqueeze(torch.tensor(labels),1)
            labels = labels.contiguous().view(-1, 1) 
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        
        anchor_feature = features.float()
        contrast_feature = features.float()
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),temperature)  
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  
        logits_mask = torch.scatter(
            torch.ones_like(mask),  
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask   

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask  
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) 
        
        # compute weight
        weight = 1-torch.exp(log_prob)
        
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (weight * mask * log_prob).mean(1)

        # loss
        mean_log_prob_pos = - (temperature / base_temperature) * mean_log_prob_pos
        mean_log_prob_pos = mean_log_prob_pos.view(batch_size)
        
        N_nonzeor = torch.nonzero(mask.sum(1)).shape[0]
        loss = mean_log_prob_pos.sum()/N_nonzeor
        if torch.isnan(loss):
            print("nan contrastive loss")
            loss=torch.zeros(1).to(device)          
        return {
            'loss': loss,
        }


class FocalLossHead(nn.Module):
    def __init__(self, hidden_states, num_classes, gamma=2.0, alpha=0.25):
        super().__init__()
        self.proj = nn.Linear(hidden_states, num_classes)
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, x, labels=None):
        logits = self.proj(x)
        log_probs = F.log_softmax(logits, dim=-1)
        
        if labels is not None:
            targets = F.one_hot(labels, num_classes=logits.size(-1)).float()
            loss = -self.alpha * (1 - torch.exp(log_probs)) ** self.gamma * targets * log_probs
            loss = loss.sum(dim=-1).mean()
        else:
            loss = None
        
        return {
            'logits': logits,
            'loss': loss
        }


class HierarchicalSupConHead(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07, 
                 label_weight=0.1, sublabel_weight=1.0):
        super(HierarchicalSupConHead, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.label_weight = label_weight
        self.sublabel_weight = sublabel_weight

    def forward(self, features, labels=None, sublabels=None, mask=None):
        """
        带权重的两级分类对比学习
        
        Args:
            features: 特征向量 [batch_size, feature_dim]
            labels: 主标签 [batch_size]
            sublabels: 子标签 [batch_size] 
            mask: 可选的对比掩码
        """
        if len(features.shape) == 3:
            features = features.squeeze(1)
            
        device = features.device
        batch_size = features.shape[0]
        
        if labels is None or sublabels is None:
            raise ValueError('Both labels and sublabels are required')
        
        # 确保标签格式正确
        labels = labels.contiguous().view(-1, 1)
        sublabels = sublabels.contiguous().view(-1, 1)
        
        # 创建掩码
        label_mask = torch.eq(labels, labels.T).float().to(device)
        sublabel_mask = torch.eq(sublabels, sublabels.T).float().to(device)
        # 子标签对比只在相同主标签内进行
        sublabel_mask = sublabel_mask * label_mask
        
        # 计算相似度
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T), 
            self.temperature
        )
        
        # 数值稳定性
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # 排除自对比
        logits_mask = torch.scatter(
            torch.ones_like(label_mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        
        # 计算概率
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        weight = 1 - torch.exp(log_prob)
        
        # 主标签损失
        label_contrastive_mask = label_mask * logits_mask
        label_pos_pairs = label_contrastive_mask.sum(1)
        label_pos_pairs = torch.where(label_pos_pairs < 1e-6, 1, label_pos_pairs)
        mean_log_prob_pos_label = (weight * label_contrastive_mask * log_prob).sum(1) / label_pos_pairs
        label_loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos_label
        
        # 子标签损失
        sublabel_contrastive_mask = sublabel_mask * logits_mask  
        sublabel_pos_pairs = sublabel_contrastive_mask.sum(1)
        sublabel_pos_pairs = torch.where(sublabel_pos_pairs < 1e-6, 1, sublabel_pos_pairs)
        mean_log_prob_pos_sublabel = (weight * sublabel_contrastive_mask * log_prob).sum(1) / sublabel_pos_pairs
        sublabel_loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos_sublabel
        
        # 平均损失
        valid_label_samples = (label_contrastive_mask.sum(1) > 0).float()
        valid_sublabel_samples = (sublabel_contrastive_mask.sum(1) > 0).float()
        
        if valid_label_samples.sum() > 0:
            label_loss = (label_loss * valid_label_samples).sum() / valid_label_samples.sum()
        else:
            label_loss = features.sum() * 0.0  # 返回一个零张量
            
        if valid_sublabel_samples.sum() > 0:
            sublabel_loss = (sublabel_loss * valid_sublabel_samples).sum() / valid_sublabel_samples.sum()
        else:
            sublabel_loss = features.sum() * 0.0
        
        # 处理NaN
        if torch.isnan(label_loss):
            label_loss = features.sum() * 0.0
        if torch.isnan(sublabel_loss):
            sublabel_loss = features.sum() * 0.0
        
        # 加权组合
        total_loss = (self.label_weight * label_loss + 
                      self.sublabel_weight * sublabel_loss)
        
        return {
            'loss': total_loss,
        }
    
    def forward_with_gathered_features(
        self, 
        local_features,
        gathered_features, 
        gathered_labels, 
        gathered_sublabels, 
        start_idx, 
        end_idx
    ):
        """
        使用聚合特征的层次化监督对比学习
        """

        if len(gathered_features.shape) == 3:
            gathered_features = gathered_features.squeeze(1)
            
        device = gathered_features.device
        
        # 提取本地特征（保持梯度）
        local_labels = gathered_labels[start_idx:end_idx]
        local_sublabels = gathered_sublabels[start_idx:end_idx]
        
        batch_size = local_features.shape[0]
        global_batch_size = gathered_features.shape[0]
        
        # 确保标签格式正确
        local_labels = local_labels.contiguous().view(-1, 1)
        local_sublabels = local_sublabels.contiguous().view(-1, 1)
        gathered_labels = gathered_labels.contiguous().view(-1, 1)
        gathered_sublabels = gathered_sublabels.contiguous().view(-1, 1)
        
        # 创建掩码
        label_mask = torch.eq(local_labels, gathered_labels.T).float().to(device)
        sublabel_mask = torch.eq(local_sublabels, gathered_sublabels.T).float().to(device)
        # 子标签对比只在相同主标签内进行
        sublabel_mask = sublabel_mask * label_mask
        
        # 计算相似度：local_features @ gathered_features.T
        anchor_dot_contrast = torch.div(
            torch.matmul(local_features, gathered_features.T), 
            self.temperature
        )
        
        # 数值稳定性
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # 排除自对比 - 创建logits_mask
        logits_mask = torch.ones(batch_size, global_batch_size, device=device)
        for i in range(batch_size):
            global_idx = start_idx + i
            if global_idx < global_batch_size:
                logits_mask[i, global_idx] = 0
        
        # 计算概率和权重
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        weight = 1 - torch.exp(log_prob)  # 与forward方法一致的权重计算
        
        # 主标签损失
        label_contrastive_mask = label_mask * logits_mask
        label_pos_pairs = label_contrastive_mask.sum(1)
        label_pos_pairs = torch.where(label_pos_pairs < 1e-6, 1, label_pos_pairs)
        mean_log_prob_pos_label = (weight * label_contrastive_mask * log_prob).sum(1) / label_pos_pairs
        label_loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos_label
        
        # 子标签损失
        sublabel_contrastive_mask = sublabel_mask * logits_mask  
        sublabel_pos_pairs = sublabel_contrastive_mask.sum(1)
        sublabel_pos_pairs = torch.where(sublabel_pos_pairs < 1e-6, 1, sublabel_pos_pairs)
        mean_log_prob_pos_sublabel = (weight * sublabel_contrastive_mask * log_prob).sum(1) / sublabel_pos_pairs
        sublabel_loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos_sublabel
        
        # 平均损失 - 与forward方法一致
        valid_label_samples = (label_contrastive_mask.sum(1) > 0).float()
        valid_sublabel_samples = (sublabel_contrastive_mask.sum(1) > 0).float()
        
        if valid_label_samples.sum() > 0:
            label_loss = (label_loss * valid_label_samples).sum() / valid_label_samples.sum()
        else:
            label_loss = local_features.sum() * 0
            
        if valid_sublabel_samples.sum() > 0:
            sublabel_loss = (sublabel_loss * valid_sublabel_samples).sum() / valid_sublabel_samples.sum()
        else:
            sublabel_loss = local_features.sum() * 0
        
        # 处理NaN
        if torch.isnan(label_loss):
            label_loss = local_features.sum() * 0
        if torch.isnan(sublabel_loss):
            sublabel_loss = local_features.sum() * 0
        
        # 加权组合 - 与forward方法一致
        total_loss = (self.label_weight * label_loss + 
                    self.sublabel_weight * sublabel_loss)
        
        return {
            'loss': total_loss,
        }