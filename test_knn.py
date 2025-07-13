import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
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
    train_data_path: str = 'test_subdatasets'  # 用于构建KNN的训练数据
    batch_size: int = 128
    model_path: str = 'output_foccon_only/model_3000.pth'
    output_dir: str = 'test_results_knn'
    
    # KNN Configurations
    k_neighbors: int = 5
    samples_per_class: int = 100  # 每个类别用于构建KNN的样本数
    
    # Model Configurations (需要与训练时保持一致)
    encoder_cfg_path: str = '/cpfs02/shared/speechllm/liuzhan/workspace_sci/icefall_general_encoder/egs/general_audio_encoder/mtl/zipformer_audio_encoder/whisper-encoder/whisper-encoder-146M'
    encoder_ckpt_path: str = '/cpfs02/shared/speechllm/liuzhan/workspace_sci/icefall_general_encoder/egs/general_audio_encoder/mtl/zipformer_audio_encoder/exp-ds-xlarge-lr-0.02-full-en-zh-audio-multi-kd-time-mask-ratio-2.0-shar/iter-352000-avg-2.pt'
    feature_dim: int = 1024
    num_classes: int = 4

def build_dataloader(config, data_path, batch_size=None):
    """构建数据加载器"""
    if batch_size is None:
        batch_size = config.batch_size
        
    ds = FolderDataset(data_path, transform=None)
    
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

    dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
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


def build_knn_classifier(train_features, train_labels, k_neighbors=5):
    """构建KNN分类器"""
    print(f"Building KNN classifier with k={k_neighbors}")
    print(f"Training data shape: {train_features.shape}")
    print(f"Training label distribution: {torch.bincount(train_labels)}")
    
    # 使用sklearn的KNN分类器
    knn = KNeighborsClassifier(n_neighbors=k_neighbors, metric='cosine')
    knn.fit(train_features.numpy(), train_labels.numpy())
    
    return knn


def evaluate_knn_classification(model, knn_classifier, test_dataloader, device, class_names=None):
    """评估KNN分类性能 - 一边提取特征一边预测"""
    all_predictions = []
    all_labels = []
    all_test_features = []  # 可选：如果需要保存测试特征
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating KNN"):
            inputs = batch['inputs'].to(device)
            class_idx = batch['class_idx'].to(device)
            pad_mask = batch['pad_mask'].to(device)
            
            # 提取测试特征
            test_features = model.encode(inputs, pad_mask)
            
            # 使用KNN进行预测
            batch_predictions = knn_classifier.predict(test_features.cpu().numpy())
            
            all_predictions.append(batch_predictions)
            all_labels.append(class_idx.cpu().numpy())
            all_test_features.append(test_features.cpu())
    
    # 合并所有批次的结果
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_test_features = torch.cat(all_test_features, dim=0)
    
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
        'test_features': all_test_features,
        'classification_report': report,
        'confusion_matrix': cm,
        'class_names': class_names
    }


def extract_balanced_features_efficient(model, dataloader, device, samples_per_class=100):
    """从每个类别提取指定数量的样本特征 - 高效版本"""
    class_samples = {}  # 存储每个类别的样本
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting KNN training samples"):
            # 检查是否所有类别都收集够了
            if len(class_samples) > 0:
                all_classes_full = all(len(class_samples.get(cls, [])) >= samples_per_class 
                                     for cls in class_samples.keys())
                if all_classes_full and len(class_samples) >= 4:  # 假设有4个类别
                    print("All classes have enough samples, stopping early...")
                    break
            
            inputs = batch['inputs'].to(device)
            class_idx = batch['class_idx'].to(device)
            subclass_idx = batch['subclass_idx'].to(device)
            pad_mask = batch['pad_mask'].to(device)
            
            # 检查这个batch中哪些样本是我们需要的
            needed_indices = []
            for i in range(len(class_idx)):
                cls = int(class_idx[i].item())
                if cls not in class_samples:
                    class_samples[cls] = {'features': [], 'labels': []}
                
                if len(class_samples[cls]['features']) < samples_per_class:
                    needed_indices.append(i)
            
            # 如果这个batch中没有需要的样本，跳过特征提取
            if not needed_indices:
                continue
            
            # 只对需要的样本提取特征
            if len(needed_indices) == len(inputs):
                # 如果整个batch都需要，直接提取
                embeds = model.encode(inputs, pad_mask)
            else:
                # 只提取需要的样本
                needed_inputs = inputs[needed_indices]
                needed_pad_mask = pad_mask[needed_indices]
                embeds = model.encode(needed_inputs, needed_pad_mask)
            
            # 存储特征
            embed_idx = 0
            for i in needed_indices:
                cls = int(class_idx[i].item())
                if len(class_samples[cls]['features']) < samples_per_class:
                    class_samples[cls]['features'].append(embeds[embed_idx].cpu())
                    class_samples[cls]['labels'].append(class_idx[i].cpu())
                    embed_idx += 1
    
    # 整理数据
    features = []
    labels = []
    
    for cls in sorted(class_samples.keys()):
        n_samples = min(samples_per_class, len(class_samples[cls]['features']))
        print(f"Class {cls}: collected {n_samples} samples for KNN training")
        
        features.extend(class_samples[cls]['features'][:n_samples])
        labels.extend(class_samples[cls]['labels'][:n_samples])
    
    return torch.stack(features), torch.stack(labels)


def save_results(results, train_features, train_labels, config):
    """保存测试结果"""
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 保存分类报告
    with open(os.path.join(config.output_dir, 'knn_classification_report.json'), 'w') as f:
        json.dump(results['classification_report'], f, indent=2)
    
    # 保存混淆矩阵图
    plot_confusion_matrix(
        results['confusion_matrix'], 
        results['class_names'],
        os.path.join(config.output_dir, 'knn_confusion_matrix.png')
    )
    
    # 保存特征分布图
    plot_feature_distribution(
        results['test_features'], torch.tensor(results['labels']), results['class_names'],
        os.path.join(config.output_dir, 'knn_feature_distribution.png')
    )
    
    # 保存特征和预测结果
    np.save(os.path.join(config.output_dir, 'train_features.npy'), train_features.numpy())
    np.save(os.path.join(config.output_dir, 'train_labels.npy'), train_labels.numpy())
    np.save(os.path.join(config.output_dir, 'test_features.npy'), results['test_features'].numpy())
    np.save(os.path.join(config.output_dir, 'test_labels.npy'), results['labels'])
    np.save(os.path.join(config.output_dir, 'knn_predictions.npy'), results['predictions'])
    
    # 保存测试总结
    summary = {
        'knn_k': config.k_neighbors,
        'samples_per_class_for_training': config.samples_per_class,
        'overall_accuracy': float(results['accuracy']),
        'num_test_samples': len(results['labels']),
        'num_classes': len(results['class_names']),
        'class_names': results['class_names'],
        'per_class_accuracy': {
            class_name: results['classification_report'][class_name]['precision']
            for class_name in results['class_names']
        }
    }
    
    with open(os.path.join(config.output_dir, 'knn_test_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    config = TestConfig()
    
    # 检查模型文件是否存在
    if not os.path.exists(config.model_path):
        print(f"Error: Model file {config.model_path} not found!")
        return
    
    # 检查数据路径是否存在
    if not os.path.exists(config.test_data_path):
        print(f"Error: Test data path {config.test_data_path} not found!")
        return
    
    if not os.path.exists(config.train_data_path):
        print(f"Error: Train data path {config.train_data_path} not found!")
        return
    
    print("Loading model...")
    model = load_model(config, device)
    
    print("Loading training data for KNN...")
    train_dataloader, train_dataset = build_dataloader(config, config.train_data_path)
    
    print("Loading test data...")
    test_dataloader, test_dataset = build_dataloader(config, config.test_data_path)
    
    # 获取类别名称
    class_names = [f'Class_{i}' for i in range(config.num_classes)]
    if hasattr(test_dataset, 'class_to_idx'):
        class_names = list(test_dataset.class_to_idx.keys())
    
    print("Extracting training features for KNN...")
    train_features, train_labels = extract_balanced_features_efficient(
        model, train_dataloader, device, config.samples_per_class
    )
    
    print("Building KNN classifier...")
    knn_classifier = build_knn_classifier(train_features, train_labels, config.k_neighbors)
    
    print("Evaluating KNN classification performance...")
    results = evaluate_knn_classification(model, knn_classifier, test_dataloader, device, class_names)
    
    # 打印结果摘要
    print("\n" + "="*50)
    print("KNN TEST RESULTS SUMMARY")
    print("="*50)
    print(f"KNN k-neighbors: {config.k_neighbors}")
    print(f"Training samples per class: {config.samples_per_class}")
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
    print("  - knn_classification_report.json")
    print("  - knn_confusion_matrix.png")
    print("  - knn_feature_distribution.png")
    print("  - train_features.npy")
    print("  - train_labels.npy")
    print("  - test_features.npy")
    print("  - test_labels.npy")
    print("  - knn_predictions.npy")
    print("  - knn_test_summary.json")

if __name__ == '__main__':
    main()