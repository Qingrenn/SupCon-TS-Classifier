import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from data import FolderDataset
from model import EncoderForPretraining
from tqdm import tqdm
from dataclasses import dataclass

@dataclass
class TestConfig:
    # Test Configurations
    test_data_path: str = 'test_datasets'
    batch_size: int = 128
    model_path: str = 'output_foccls_only/model_3000.pth'
    output_dir: str = 'test_results'
    
    # Model Configurations (需要与训练时保持一致)
    encoder_cfg_path: str = '/cpfs02/shared/speechllm/liuzhan/workspace_sci/icefall_general_encoder/egs/general_audio_encoder/mtl/zipformer_audio_encoder/whisper-encoder/whisper-encoder-146M'
    encoder_ckpt_path: str = '/cpfs02/shared/speechllm/liuzhan/workspace_sci/icefall_general_encoder/egs/general_audio_encoder/mtl/zipformer_audio_encoder/exp-ds-xlarge-lr-0.02-full-en-zh-audio-multi-kd-time-mask-ratio-2.0-shar/iter-352000-avg-2.pt'
    feature_dim: int = 1024
    num_classes: int = 4

def build_test_dataloader(config):
    """构建测试数据加载器"""
    ds = FolderDataset(config.test_data_path, transform=None)
    
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

    dl = DataLoader(ds, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=False)
    return dl, ds

def load_model(config, device):
    """加载训练好的模型"""
    model = EncoderForPretraining(config)
    
    # 加载模型权重
    checkpoint = torch.load(config.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model

def extract_features(model, dataloader, device):
    """提取特征向量"""
    features = []
    labels = []
    subclass_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            inputs = batch['inputs'].to(device)
            class_idx = batch['class_idx'].to(device)
            subclass_idx = batch['subclass_idx'].to(device)
            pad_mask = batch['pad_mask'].to(device)
            
            # 提取特征
            embeds = model.encode(inputs, pad_mask)
            
            features.append(embeds.cpu())
            labels.append(class_idx.cpu())
            subclass_labels.append(subclass_idx.cpu())
    
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    subclass_labels = torch.cat(subclass_labels, dim=0)
    
    return features, labels, subclass_labels

def evaluate_classification(model, dataloader, device, class_names=None):
    """评估分类性能"""
    all_predictions = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = batch['inputs'].to(device)
            class_idx = batch['class_idx'].to(device)
            pad_mask = batch['pad_mask'].to(device)
            
            # 获取分类结果
            output = model(inputs, pad_mask=pad_mask)
            logits = output['logits']
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.append(predictions.cpu())
            all_labels.append(class_idx.cpu())
            all_logits.append(logits.cpu())
    
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_logits = torch.cat(all_logits, dim=0).numpy()
    
    # 计算准确率
    accuracy = (all_predictions == all_labels).mean()
    
    # 生成分类报告
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(len(np.unique(all_labels)))]
    
    report = classification_report(
        all_labels, all_predictions, 
        target_names=class_names, 
        output_dict=True
    )
    
    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    
    return {
        'accuracy': accuracy,
        'predictions': all_predictions,
        'labels': all_labels,
        'logits': all_logits,
        'classification_report': report,
        'confusion_matrix': cm,
        'class_names': class_names
    }

def plot_confusion_matrix(cm, class_names, save_path):
    """绘制并保存混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_distribution(features, labels, class_names, save_path):
    """使用t-SNE可视化特征分布"""
    try:
        from sklearn.manifold import TSNE
        
        # 如果样本太多，随机采样
        if len(features) > 5000:
            indices = np.random.choice(len(features), 5000, replace=False)
            features_subset = features[indices]
            labels_subset = labels[indices]
        else:
            features_subset = features
            labels_subset = labels
        
        # t-SNE降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features_subset.numpy())
        
        # 绘制
        plt.figure(figsize=(12, 10))
        colors = plt.cm.Set1(np.linspace(0, 1, len(class_names)))
        
        for i, class_name in enumerate(class_names):
            mask = labels_subset == i
            if mask.sum() > 0:
                plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                           c=[colors[i]], label=class_name, alpha=0.6, s=20)
        
        plt.title('Feature Distribution (t-SNE)')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except ImportError:
        print("Warning: sklearn not available, skipping t-SNE visualization")

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    config = TestConfig()
    
    # 检查模型文件是否存在
    if not os.path.exists(config.model_path):
        print(f"Error: Model file {config.model_path} not found!")
        return
    
    # 检查测试数据是否存在
    if not os.path.exists(config.test_data_path):
        print(f"Error: Test data path {config.test_data_path} not found!")
        return
    
    print("Loading test data...")
    test_dataloader, test_dataset = build_test_dataloader(config)
    
    # 获取类别名称
    class_names = [f'Class_{i}' for i in range(config.num_classes)]
    if hasattr(test_dataset, 'class_to_idx'):
        class_names = list(test_dataset.class_to_idx.keys())
    
    print("Loading model...")
    model = load_model(config, device)
    
    # print("Extracting features...")
    # features, labels, subclass_labels = extract_features(model, test_dataloader, device)
    
    print("Evaluating classification performance...")
    results = evaluate_classification(model, test_dataloader, device, class_names)
    
    # 打印结果摘要
    print("\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)
    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    print(f"Number of test samples: {len(results['labels'])}")
    print(f"Number of classes: {len(class_names)}")
    
    print("\nPer-class Results:")
    for i, class_name in enumerate(class_names):
        if class_name in results['classification_report']:
            precision = results['classification_report'][class_name]['precision']
            recall = results['classification_report'][class_name]['recall']
            f1 = results['classification_report'][class_name]['f1-score']
            support = results['classification_report'][class_name]['support']
            print(f"  {class_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, N={support}")
    
    print(f"\nResults saved to: {config.output_dir}")
    print("Files generated:")
    print("  - classification_report.json")
    print("  - confusion_matrix.png")
    print("  - feature_distribution.png")
    print("  - features.npy")
    print("  - labels.npy")
    print("  - predictions.npy")
    print("  - logits.npy")
    print("  - test_summary.json")

if __name__ == '__main__':
    main()