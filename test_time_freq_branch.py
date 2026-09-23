import torch
import torch.nn as nn
import numpy as np
import h5py
import os
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from Models.PyTorchModels_backup_before_lightweight import FeaAndEmg_se_PyTorch, LinearResidualBlock

class SimpleTimeFreqModel(nn.Module):
    """简化的时频分支模型：直接使用PCA特征"""
    def __init__(self, emg_channels, time_steps, feature_dim, num_classes, dropout_rate=0.4):
        super().__init__()
        # 简化的空间分支：直接处理EMG数据
        self.space_branch = nn.Sequential(
            nn.Linear(emg_channels * time_steps, 128),  # 展平EMG数据
            nn.BatchNorm1d(128),
            nn.PReLU()
        )
        
        # 时频分支：直接使用PCA特征，不进行神经网络处理
        self.time_freq_branch = None  # 不使用神经网络
        
        # 门控和分类器
        self.gate = nn.Sequential(
            nn.Linear(192, 2),  # 128(空间) + 64(PCA) = 192
            nn.Softmax(dim=-1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(192, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, feature_data, emg_data, return_gate_weights=False):
        # 空间特征分支：展平EMG数据
        batch_size = emg_data.size(0)
        emg_flat = emg_data.view(batch_size, -1)  # 展平为 (batch, channels*time)
        space_features = self.space_branch(emg_flat)
        
        # 时频特征：直接使用PCA特征
        time_freq_features = feature_data  # 64维PCA特征
        
        # 融合特征
        combined_features = torch.cat([space_features, time_freq_features], dim=1)
        
        # 门控权重
        gate_weights = self.gate(combined_features)
        
        # 加权融合
        gated_space_features = space_features * gate_weights[:, 0].unsqueeze(1)
        gated_time_freq_features = time_freq_features * gate_weights[:, 1].unsqueeze(1)
        
        final_features = torch.cat([gated_space_features, gated_time_freq_features], dim=1)
        
        # 分类
        output = self.classifier(final_features)
        
        if return_gate_weights:
            return output, gate_weights
        else:
            return output

def test_model_comparison():
    """对比测试：有时频分支 vs 无时频分支"""
    
    # 设置参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subject_id = 5
    root_data = 'D:/DB4'
    
    # 加载数据
    emg_file = os.path.join(root_data, f'data/restimulus/reSegEnhance/DB4_s{subject_id}SegEnhance.h5')
    feature_file = os.path.join(root_data, f'data/restimulus/Fea/DB4_s{subject_id}feaEnhance.h5')
    
    with h5py.File(emg_file, 'r') as f:
        emg_data = f['emg'][:]
        labels = f['label'][:]
        rep_arr = f['rep'][:]
    
    with h5py.File(feature_file, 'r') as f:
        feature_data = f['features'][:]
    
    # 简单的数据分割（使用重复次数2作为验证集）
    train_mask = rep_arr != 2
    val_mask = rep_arr == 2
    
    emg_train, emg_val = emg_data[train_mask], emg_data[val_mask]
    feature_train, feature_val = feature_data[train_mask], feature_data[val_mask]
    label_train, label_val = labels[train_mask], labels[val_mask]
    
    # 创建数据集
    class SimpleDataset(Dataset):
        def __init__(self, emg_data, feature_data, labels):
            # 确保EMG数据是4维的 (batch, channels, time, 1)
            if len(emg_data.shape) == 3:  # (batch, time, channels)
                emg_data = np.transpose(emg_data, (0, 2, 1))  # (batch, channels, time)
            emg_data = emg_data[..., np.newaxis]  # (batch, channels, time, 1)
            
            self.emg_data = torch.FloatTensor(emg_data)
            self.feature_data = torch.FloatTensor(feature_data)
            self.labels = torch.LongTensor(labels)
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            return self.feature_data[idx], self.emg_data[idx], self.labels[idx]
    
    train_dataset = SimpleDataset(emg_train, feature_train, label_train)
    val_dataset = SimpleDataset(emg_val, feature_val, label_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 测试两个模型
    models = {
        'with_time_freq_branch': FeaAndEmg_se_PyTorch(12, 400, 64, np.max(label_train)+1),
        'without_time_freq_branch': SimpleTimeFreqModel(12, 400, 64, np.max(label_train)+1)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n测试模型: {model_name}")
        model = model.to(device)
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"参数量: {total_params:,}")
        
        # 简单训练（5个epoch）
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(5):
            train_loss = 0.0
            for feature_batch, emg_batch, label_batch in train_loader:
                feature_batch, emg_batch, label_batch = feature_batch.to(device), emg_batch.to(device), label_batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(feature_batch, emg_batch)
                loss = criterion(outputs, label_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            print(f"Epoch {epoch+1}, Loss: {train_loss/len(train_loader):.4f}")
        
        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for feature_batch, emg_batch, label_batch in val_loader:
                feature_batch, emg_batch, label_batch = feature_batch.to(device), emg_batch.to(device), label_batch.to(device)
                outputs = model(feature_batch, emg_batch)
                _, predicted = torch.max(outputs.data, 1)
                val_total += label_batch.size(0)
                val_correct += (predicted == label_batch).sum().item()
        
        accuracy = 100.0 * val_correct / val_total
        results[model_name] = {
            'accuracy': accuracy,
            'params': total_params
        }
        print(f"验证准确率: {accuracy:.2f}%")
    
    # 打印对比结果
    print("\n" + "="*50)
    print("对比结果:")
    print("="*50)
    
    for model_name, result in results.items():
        print(f"{model_name}:")
        print(f"  准确率: {result['accuracy']:.2f}%")
        print(f"  参数量: {result['params']:,}")
    
    # 计算差异
    acc_diff = results['with_time_freq_branch']['accuracy'] - results['without_time_freq_branch']['accuracy']
    param_diff = results['with_time_freq_branch']['params'] - results['without_time_freq_branch']['params']
    
    print(f"\n差异分析:")
    print(f"准确率差异: {acc_diff:.2f}%")
    print(f"参数量差异: {param_diff:,}")
    
    if acc_diff > 1.0:
        print("结论: 时频分支神经网络有用！")
    elif acc_diff < -1.0:
        print("结论: 时频分支神经网络有害！")
    else:
        print("结论: 时频分支神经网络效果不明显")

if __name__ == '__main__':
    test_model_comparison() 