import torch
import torch.nn as nn
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import h5py
sys.path.append('.')

# 导入模型和数据处理函数
from Models.PyTorchModels_improved_with_ablation import FeaAndEmg_se_PyTorch
from torch.utils.data import DataLoader, Dataset

# 定义EMG数据集类
class EMGDataset(Dataset):
    def __init__(self, emg_data, feature_data, labels):
        self.emg_data = torch.FloatTensor(emg_data)
        self.feature_data = torch.FloatTensor(feature_data)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.emg_data[idx], self.feature_data[idx], self.labels[idx]

def load_and_preprocess_data(subject_id=1):
    """加载并预处理指定受试者的数据"""
    try:
        # 构建数据文件路径
        root_data = 'D:/DB2'  # 根据实际路径调整
        emg_file = os.path.join(root_data, f'data/restimulus/reSegEnhance/DB2_s{subject_id}SegEnhance.h5')
        feature_file = os.path.join(root_data, f'data/restimulus/Fea/DB2_s{subject_id}feaEnhance.h5')
        
        # 检查文件是否存在
        if not os.path.exists(emg_file) or not os.path.exists(feature_file):
            print(f"错误: 数据文件不存在")
            return None
        
        # 加载数据
        print(f"加载受试者 {subject_id} 的数据...")
        with h5py.File(emg_file, 'r') as f:
            emg_data = f['emg'][:]
            labels = f['label'][:]
            rep_arr = f['rep'][:]
        
        with h5py.File(feature_file, 'r') as f:
            feature_data = f['features'][:]
        
        # 按重复次数划分数据集
        train_reps = [1, 3, 4, 6]  # 训练集
        test_reps = [5]  # 测试集
        vali_reps = [2]  # 验证集
        
        # 获取测试集索引
        test_indices = np.concatenate([np.where(rep_arr == rep)[0] for rep in test_reps])
        # 获取验证集索引
        vali_indices = np.concatenate([np.where(rep_arr == rep)[0] for rep in vali_reps])
        # 获取训练集索引
        train_indices = np.concatenate([np.where(rep_arr == rep)[0] for rep in train_reps])
        
        # 提取数据
        emg_train = emg_data[train_indices]
        emg_vali = emg_data[vali_indices]
        emg_test = emg_data[test_indices]
        
        feature_train = feature_data[train_indices]
        feature_vali = feature_data[vali_indices]
        feature_test = feature_data[test_indices]
        
        label_train = labels[train_indices]
        label_vali = labels[vali_indices]
        label_test = labels[test_indices]
        
        # 重塑EMG数据为(B, C, T, 1)
        emg_train = emg_train.reshape(emg_train.shape[0], emg_train.shape[1], emg_train.shape[2], 1)
        emg_vali = emg_vali.reshape(emg_vali.shape[0], emg_vali.shape[1], emg_vali.shape[2], 1)
        emg_test = emg_test.reshape(emg_test.shape[0], emg_test.shape[1], emg_test.shape[2], 1)
        
        print(f"数据集大小: 训练集={emg_train.shape[0]}, 验证集={emg_vali.shape[0]}, 测试集={emg_test.shape[0]}")
        
        return {
            'emg_train': emg_train,
            'emg_vali': emg_vali,
            'emg_test': emg_test,
            'feature_train': feature_train,
            'feature_vali': feature_vali,
            'feature_test': feature_test,
            'label_train': label_train,
            'label_vali': label_vali,
            'label_test': label_test,
            'emg_channels': emg_train.shape[1],
            'time_steps': emg_train.shape[2],
            'feature_dim': feature_train.shape[1],
            'num_classes': len(np.unique(labels))
        }
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None

def evaluate_model(model, test_loader, device):
    """评估模型性能"""
    model.eval()
    all_preds = []
    all_labels = []
    all_gate_weights = []
    
    with torch.no_grad():
        for emg_batch, feature_batch, labels_batch in test_loader:
            emg_batch, feature_batch = emg_batch.to(device), feature_batch.to(device)
            outputs, gate_weights = model(feature_batch, emg_batch, return_gate_weights=True)
            
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
            all_gate_weights.append(gate_weights.cpu().numpy())
    
    # 计算平均门控权重
    all_gate_weights = np.concatenate(all_gate_weights, axis=0)
    avg_gate_weights = np.mean(all_gate_weights, axis=0)
    
    # 计算准确率
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 生成分类报告
    report = classification_report(all_labels, all_preds, output_dict=True)
    
    return {
        'accuracy': accuracy,
        'gate_weights': avg_gate_weights,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': all_preds,
        'true_labels': all_labels
    }

def test_on_real_data():
    """在真实EMG数据上测试改进后的模型"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    subject_id = 1  # 可以修改为其他受试者ID
    data = load_and_preprocess_data(subject_id)
    if data is None:
        return
    
    # 创建数据加载器
    batch_size = 32
    test_dataset = EMGDataset(data['emg_test'], data['feature_test'], data['label_test'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    model = FeaAndEmg_se_PyTorch(
        emg_channels=data['emg_channels'],
        time_steps=data['time_steps'],
        feature_dim=data['feature_dim'],
        num_classes=data['num_classes'],
        dropout_rate=0.3
    ).to(device)
    
    # 评估模型
    print("\n评估模型性能...")
    results = evaluate_model(model, test_loader, device)
    
    # 打印结果
    print(f"\n测试准确率: {results['accuracy']*100:.2f}%")
    print(f"门控权重: 空间特征={results['gate_weights'][0]*100:.2f}%, 时频特征={results['gate_weights'][1]*100:.2f}%")
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(results['confusion_matrix'], annot=False, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f'confusion_matrix_subject_{subject_id}.png')
    print(f"混淆矩阵已保存到 'confusion_matrix_subject_{subject_id}.png'")
    
    # 计算每个类别的F1分数
    f1_scores = [results['classification_report'][str(i)]['f1-score'] for i in range(data['num_classes'])]
    
    # 绘制F1分数分布
    plt.figure(figsize=(12, 6))
    plt.bar(range(data['num_classes']), f1_scores)
    plt.title('F1 Score for Each Class')
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.savefig(f'f1_scores_subject_{subject_id}.png')
    print(f"F1分数分布已保存到 'f1_scores_subject_{subject_id}.png'")
    
    print("\n测试完成!")

if __name__ == "__main__":
    test_on_real_data() 