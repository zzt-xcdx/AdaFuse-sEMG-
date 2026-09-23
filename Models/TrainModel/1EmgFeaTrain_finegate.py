import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import h5py
import numpy as np
import matplotlib.pyplot as plt
from Models.PyTorchModels_improved_beifen import ImprovedFeaAndEmg_se_PyTorch
from Util.function import get_threeSet
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from tqdm import tqdm
import matplotlib

# 改进字体设置，支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.titlesize'] = 14
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 10

# 设置随机种子
np.random.seed(123)
torch.manual_seed(123)

# 数据集配置
dataset = 'DB4'  # 可以改为 'DB2' 或 'DB4'





if dataset == 'DB4':
    root_data = 'D:/DB4'
    subject_range = range(7, 9)  # 只训练受试者7和8
else:  # DB2
    root_data = 'D:/DB2'
    subject_range = range(1, 41)  # DB2有40个被试，从1开始

# 结果保存目录
RESULTS_DIR = f'results_report_finegate_{dataset}_enhanced_subjects_7_8'
os.makedirs(RESULTS_DIR, exist_ok=True)

# 早停类
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = model.state_dict().copy()
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

class EMGDataset(Dataset):
    def __init__(self, emg_data, feature_data, labels):
        # 支持 3 维 (N,T,C)/(N,C,T) 或 4 维 (N,C,T,1)
        if len(emg_data.shape) == 4:
            # (N, C, T, 1) → (N, C, T)
            emg_np = emg_data[..., 0]
        elif len(emg_data.shape) == 3:
            # 自动判断哪个轴是通道
            if emg_data.shape[-1] == 16:          # (N, T, C)
                emg_np = np.transpose(emg_data, (0, 2, 1))  # (N, C, T)
            else:                                 # 已是 (N, C, T)
                emg_np = emg_data
        else:
            raise ValueError(f"不支持的 EMG 数据维度: {emg_data.shape}")

        # === 逐窗口 z-score 归一化 ===
        mean = emg_np.mean(axis=2, keepdims=True)              # (N, C, 1)
        std  = emg_np.std(axis=2, keepdims=True) + 1e-6
        emg_np = (emg_np - mean) / std                         # (N, C, T)

        # 加回最后的 1 维，变成 (N, C, T, 1)
        emg_np = emg_np[..., np.newaxis]

        # 转成张量
        self.emg_data = torch.FloatTensor(emg_np)
        self.feature_data = torch.FloatTensor(feature_data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.emg_data[idx], self.feature_data[idx], self.labels[idx]

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 完整增强模型的训练参数
    batch_size = 32
    dropout_rate = 0.4  # 保持适中的dropout
    epochs = 150
    early_stopping_patience = 20  # 早停耐心值
    
    subject_accuracies = []
    all_subject_gate_weights = {}
    
    for subject_id in subject_range:
        print(f"\n处理受试者 {subject_id}...")
        try:
            emg_file = os.path.join(root_data, f'data/restimulus/reSegEnhance/{dataset}_s{subject_id}SegEnhance.h5')
            feature_file = os.path.join(root_data, f'data/restimulus/Fea/{dataset}_s{subject_id}feaEnhance.h5')
            if not os.path.exists(emg_file):
                print(f"错误: EMG文件未找到: {emg_file}")
                continue
            if not os.path.exists(feature_file):
                print(f"错误: 特征文件未找到: {feature_file}")
                continue
            print("加载数据...")
            with h5py.File(emg_file, 'r') as f:
                emg_data = f['emg'][:]
                labels = f['label'][:]
                rep_arr = f['rep'][:]
            with h5py.File(feature_file, 'r') as f:
                feature_data = f['features'][:]
            rep_vali = 2
            emg_train, emg_vali, emg_test, label_train, label_vali, label_test = get_threeSet(emg_data, labels, rep_arr, rep_vali)
            feature_train, feature_vali, feature_test, _, _, _ = get_threeSet(feature_data, labels, rep_arr, rep_vali)
            # 打印DB2/DB4数据维度
            print("DB2/DB4 - EMG shape:", emg_train.shape)
            print("DB2/DB4 - Feature shape:", feature_train.shape)
            print("DB2/DB4 - Num classes:", np.max(label_train)+1)
        except Exception as e:
            print(f"数据加载或分割出错: {e}")
            continue
            
        train_dataset = EMGDataset(emg_train, feature_train, label_train)
        val_dataset = EMGDataset(emg_vali, feature_vali, label_vali)
        test_dataset = EMGDataset(emg_test, feature_test, label_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 创建改进的增强模型 (使用Conv→ReLU→BN顺序)
        model = ImprovedFeaAndEmg_se_PyTorch(emg_train.shape[2], emg_train.shape[1], feature_train.shape[1], np.max(label_train)+1, dropout_rate=dropout_rate).to(device)
        
        # 打印模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        print("空间特征提取器参数：", sum(p.numel() for p in model.space_branch.spatial_extractor.parameters()))
        print("空间分支总参数：", sum(p.numel() for p in model.space_branch.parameters()))
        print("时频分支参数：", sum(p.numel() for p in model.time_freq_branch.parameters()))
        
        # 优化器
        optimizer = Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss().to(device)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
        
        # 早停机制
        early_stopping = EarlyStopping(patience=early_stopping_patience, min_delta=0.001)
        
        # 由于模型结构已修改，从头开始训练
        print(f"模型结构已更新，从头开始训练...")
        
        best_val_acc = 0.0
        best_model_state = None
        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        gate_weights_history = []
        
        for epoch in range(epochs):
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            
            for batch_idx, (fea_inputs, emg_inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
                fea_inputs, emg_inputs, labels = fea_inputs.to(device), emg_inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs, gate_weights = model(fea_inputs, emg_inputs, return_gate_weights=True)
                loss = criterion(outputs, labels)
                loss.backward()
                
                optimizer.step()
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # 统计每个batch的门控权重
                if batch_idx % 10 == 0:
                    space_weight_mean = gate_weights[:, 0].mean().item()
                    time_freq_weight_mean = gate_weights[:, 1].mean().item()
                    gate_weights_history.append((epoch + batch_idx/len(train_loader), space_weight_mean, time_freq_weight_mean))
            
            train_acc = 100.0 * train_correct / train_total
            train_losses.append(train_loss / len(train_loader))
            train_accs.append(train_acc)
            
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for fea_inputs, emg_inputs, labels in val_loader:
                    fea_inputs, emg_inputs, labels = fea_inputs.to(device), emg_inputs.to(device), labels.to(device)
                    outputs, _ = model(fea_inputs, emg_inputs, return_gate_weights=True)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = 100.0 * val_correct / val_total
            val_losses.append(val_loss / len(val_loader))
            val_accs.append(val_acc)
            
            scheduler.step(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
            
            # 早停检查
            if early_stopping(val_losses[-1], model):
                print(f"早停触发！在第 {epoch+1} 轮停止训练")
                break
        
        # 绘制训练曲线（改进字体）
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 3, 1)
        plt.plot(train_losses, 'b-', linewidth=2, label='训练损失')
        plt.plot(val_losses, 'r-', linewidth=2, label='验证损失')
        plt.title(f'训练和验证损失曲线 (被试 {subject_id})', fontsize=14, fontweight='bold')
        plt.xlabel('训练轮次', fontsize=12)
        plt.ylabel('损失值', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(train_accs, 'b-', linewidth=2, label='训练准确率')
        plt.plot(val_accs, 'r-', linewidth=2, label='验证准确率')
        plt.title(f'训练和验证准确率曲线 (被试 {subject_id})', fontsize=14, fontweight='bold')
        plt.xlabel('训练轮次', fontsize=12)
        plt.ylabel('准确率 (%)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # 绘制门控权重变化
        epochs_hist = [x[0] for x in gate_weights_history]
        space_weights = [x[1] * 100 for x in gate_weights_history]  # 转换为百分比
        time_freq_weights = [x[2] * 100 for x in gate_weights_history]  # 转换为百分比
        
        plt.subplot(1, 3, 3)
        plt.plot(epochs_hist, space_weights, 'b-', linewidth=2, label='空间分支权重')
        plt.plot(epochs_hist, time_freq_weights, 'g-', linewidth=2, label='时频分支权重')
        plt.title(f'门控权重变化 (被试 {subject_id})', fontsize=14, fontweight='bold')
        plt.xlabel('训练轮次', fontsize=12)
        plt.ylabel('权重百分比 (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'training_curves_subject_{subject_id}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存模型权重
        torch.save(best_model_state, os.path.join(RESULTS_DIR, f'best_model_subject_{subject_id}.pth'))
        print(f"最佳模型已保存到 {os.path.join(RESULTS_DIR, f'best_model_subject_{subject_id}.pth')}")
        
        # --- 开始测试 ---
        model.load_state_dict(best_model_state)
        model.eval()
        all_preds, all_labels, all_gate_weights = [], [], []
        
        with torch.no_grad():
            for fea_inputs, emg_inputs, labels in test_loader:
                fea_inputs, emg_inputs, labels = fea_inputs.to(device), emg_inputs.to(device), labels.to(device)
                outputs, gate_weights = model(fea_inputs, emg_inputs, return_gate_weights=True)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_gate_weights.append(gate_weights.cpu().numpy())
        
        test_acc = 100.0 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
        subject_accuracies.append(test_acc)
        final_gate_weights = np.mean(np.concatenate(all_gate_weights, axis=0), axis=0)
        all_subject_gate_weights[subject_id] = final_gate_weights
        
        print(f"受试者 {subject_id} 测试准确率: {test_acc:.2f}%")
        print(f"测试集门控权重: 空间分支={final_gate_weights[0]*100:.2f}%, 时频分支={final_gate_weights[1]*100:.2f}%")
        
        # 绘制混淆矩阵（改进字体）
        cm = confusion_matrix(all_labels, all_preds)
        cm_path = os.path.join(RESULTS_DIR, f'confusion_matrix_subject_{subject_id}.png')
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, aspect='auto')
        plt.title(f'混淆矩阵 - 被试 {subject_id}', fontsize=16, fontweight='bold')
        cbar = plt.colorbar()
        cbar.set_label('样本数量', fontsize=12)
        plt.ylabel('真实标签', fontsize=12)
        plt.xlabel('预测标签', fontsize=12)
        
        # 在混淆矩阵中显示数字
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=10)
        
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        report_str = classification_report(all_labels, all_preds)
        print(report_str)
        report_path = os.path.join(RESULTS_DIR, f'classification_report_subject_{subject_id}.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_str)
        print(f"分类报告已保存到 {report_path}")
    
    # 输出总体结果
    for i, accuracy in enumerate(subject_accuracies):
        print(f"受试者 {i+1} 的准确率: {accuracy:.2f}%")
    
    if len(subject_accuracies) > 0:
        overall_accuracy = np.mean(subject_accuracies)
        accuracy_std = np.std(subject_accuracies)
        accuracy_max = np.max(subject_accuracies)
        accuracy_min = np.min(subject_accuracies)
        
        print(f"\n=== 总体评估结果 ===")
        print(f"处理的受试者数量: {len(subject_accuracies)}")
        print(f"总体分类准确度 (OA): {overall_accuracy:.2f}%")
        print(f"准确度标准差: {accuracy_std:.2f}%")
        print(f"最高准确度: {accuracy_max:.2f}%")
        print(f"最低准确度: {accuracy_min:.2f}%")
        
        avg_gate_weights_all_subjects = np.mean(list(all_subject_gate_weights.values()), axis=0)
        print(f"\n所有被试平均门控权重: 空间分支={avg_gate_weights_all_subjects[0]*100:.2f}%, 时频分支={avg_gate_weights_all_subjects[1]*100:.2f}%")
        
        results_summary = {
            'subject_accuracies': subject_accuracies,
            'overall_accuracy': overall_accuracy,
            'accuracy_std': accuracy_std,
            'accuracy_max': accuracy_max,
            'accuracy_min': accuracy_min,
            'num_subjects': len(subject_accuracies),
            'gate_weights': all_subject_gate_weights,
            'avg_gate_weights': avg_gate_weights_all_subjects
        }
        
        np.save(os.path.join(RESULTS_DIR, 'subject_accuracies.npy'), results_summary)
        print(f"结果已保存到 {os.path.join(RESULTS_DIR, 'subject_accuracies.npy')}")
    else:
        print("没有收集到任何受试者的准确度数据")

if __name__ == '__main__':
    main() 