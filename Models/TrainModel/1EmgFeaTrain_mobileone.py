import sys
import os
sys.path.append('.')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import h5py
import numpy as np
import matplotlib.pyplot as plt
# 旧版本模型已不再使用
# from Models.PyTorchModels import FeaAndEmg_se_PyTorch
from Util.function import get_threeSet
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import NAdam
from tqdm import tqdm
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

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
RESULTS_DIR = f'results_report_{dataset}'
os.makedirs(RESULTS_DIR, exist_ok=True)

def pltCurve(loss, val_loss, accuracy, val_accuracy, subject_id=None):
    epochs = range(1, len(loss) + 1)
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='Training accuracy')
    plt.plot(epochs, val_accuracy, label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    # 为每个被试生成独立的文件名
    if subject_id is not None:
        filename = os.path.join(RESULTS_DIR, f'training_curves_subject_{subject_id}.png')
    else:
        filename = os.path.join(RESULTS_DIR, 'training_curves.png')
    
    plt.savefig(filename)
    plt.close()
    print(f"训练曲线已保存到 {filename}")

class EMGDataset(Dataset):
    def __init__(self, emg_data, feature_data, labels, train=True):
        # 只处理4维 (batch, channels, time, 1)
        if len(emg_data.shape) == 4:
            self.emg_data = torch.FloatTensor(emg_data)
        else:
            raise ValueError(f"不支持的EMG数据维度: {emg_data.shape}")
        self.feature_data = torch.FloatTensor(feature_data)
        self.labels = torch.LongTensor(labels)
        self.train = train  # 区分训练和测试模式
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        feature, emg, label = self.feature_data[idx], self.emg_data[idx], self.labels[idx]
        
        # -------- 归一化 ----------
        emg = (emg - emg.mean(-2, keepdim=True)) / (emg.std(-2, keepdim=True) + 1e-6)
        feature = (feature - feature.mean()) / (feature.std() + 1e-6)
        
        # -------- 简易增强（仅训练时）---------
        if self.train:
            emg += 0.02 * torch.randn_like(emg)
            scale = torch.empty(1).uniform_(0.9, 1.1)
            emg *= scale
            
        return feature, emg, label

def emg_augment(batch, noise_std=0.02, scale_low=0.9, scale_high=1.1):
    """sEMG数据增强函数"""
    # 高斯噪声
    batch += noise_std * torch.randn_like(batch)
    # 振幅缩放
    scales = torch.empty(batch.size(0), 1, 1, 1, device=batch.device).uniform_(scale_low, scale_high)
    batch *= scales
    return batch

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience

def load_data(emg_file, feature_file):
    """加载 EMG 数据、特征数据和重复次数数组"""
    with h5py.File(emg_file, 'r') as f:
        emg_data = f['emg'][:]
        labels = f['label'][:]
        rep_arr = f['rep'][:] # 加载重复次数数组
    
    with h5py.File(feature_file, 'r') as f:
            feature_data = f['features'][:]

    # 将标签从1开始调整为从0开始
    labels = labels - 1

    return emg_data, feature_data, labels, rep_arr

def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):
    plt.figure(figsize=(20, 16), dpi=100)
    np.set_printoptions(precision=2)
    classes = list(range(len(cm)))
    iters = np.reshape([[[i, j] for j in range(len(classes))] for i in range(len(classes))], (cm.size, 2))
    for i, j in iters:
        plt.text(j, i, format(cm[i, j]))
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.title(title)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)
    plt.colorbar()
    plt.savefig(savename, format='png')
    plt.close()

def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 计算F1分数
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # 计算准确率
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    
    # 计算每个类别的指标
    class_report = classification_report(y_true, y_pred, output_dict=True)
    
    return accuracy, f1, cm, class_report

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置超参数
    batch_size = 32  # 改回原来的batch size以保持训练稳定性
    dropout_rate = 0.4
    epochs = 200
    
    # 添加正则化参数
    weight_decay = 1e-4  # L2正则化
    label_smoothing = 0.1  # 标签平滑
    
    # 存储所有受试者的结果
    subject_accuracies = []
    all_subject_gate_weights = {}
    
    # 遍历所有受试者
    for subject_id in subject_range:
        print(f"\n处理受试者 {subject_id}...")
        
        # 加载数据
        try:
            # 构建当前受试者的数据文件路径
            emg_file = os.path.join(root_data, f'data/restimulus/reSegEnhance/{dataset}_s{subject_id}SegEnhance.h5')
            feature_file = os.path.join(root_data, f'data/restimulus/Fea/{dataset}_s{subject_id}feaEnhance.h5')
            
            # 检查文件是否存在
            if not os.path.exists(emg_file):
                print(f"错误: EMG文件未找到: {emg_file}")
                continue
            if not os.path.exists(feature_file):
                print(f"错误: 特征文件未找到: {feature_file}")
                continue
            
            # 加载数据
            print("加载数据...")
            with h5py.File(emg_file, 'r') as f:
                emg_data = f['emg'][:]
                labels = f['label'][:]
                rep_arr = f['rep'][:]
            
            with h5py.File(feature_file, 'r') as f:
                feature_data = f['features'][:]
            
            # 使用get_threeSet按重复次数划分训练集、验证集和测试集
            rep_vali = 2  # 使用重复次数2作为验证集
            emg_train, emg_vali, emg_test, label_train, label_vali, label_test = get_threeSet(emg_data, labels, rep_arr, rep_vali)
            feature_train, feature_vali, feature_test, _, _, _ = get_threeSet(feature_data, labels, rep_arr, rep_vali)

            # 验证数据集划分的正确性
            train_indices = np.where(np.isin(rep_arr, [1, 3, 4, 6]))[0]
            val_indices = np.where(rep_arr == 2)[0]
            test_indices = np.where(rep_arr == 5)[0]
            
            train_val_overlap = len(set(train_indices) & set(val_indices))
            train_test_overlap = len(set(train_indices) & set(test_indices))
            val_test_overlap = len(set(val_indices) & set(test_indices))
            
            if train_val_overlap > 0 or train_test_overlap > 0 or val_test_overlap > 0:
                print("警告：数据集存在重叠，请检查划分逻辑！")
                continue

            # 标签映射，确保从0开始且连续
            all_labels = np.concatenate([label_train, label_vali, label_test])
            unique_labels = np.unique(all_labels)
            label_map = {old: new for new, old in enumerate(unique_labels)}
            label_train = np.array([label_map[x] for x in label_train])
            label_vali = np.array([label_map[x] for x in label_vali])
            label_test = np.array([label_map[x] for x in label_test])
            num_classes = len(unique_labels)

            # 获取数据维度
            emg_channels = emg_train.shape[1]
            time_steps = emg_train.shape[2]
            feature_dim = feature_train.shape[1]
            
            # 重塑EMG数据为(B, C, T, 1)
            emg_train = emg_train.reshape(emg_train.shape[0], emg_train.shape[1], emg_train.shape[2], 1)
            emg_vali = emg_vali.reshape(emg_vali.shape[0], emg_vali.shape[1], emg_vali.shape[2], 1)
            emg_test = emg_test.reshape(emg_test.shape[0], emg_test.shape[1], emg_test.shape[2], 1)
            
        except Exception as e:
            print(f"加载受试者 {subject_id} 数据时出错: {e}")
            continue
            
        # --- 开始训练 ---
        train_dataset = EMGDataset(emg_train, feature_train, label_train, train=True)
        val_dataset = EMGDataset(emg_vali, feature_vali, label_vali, train=False)
        test_dataset = EMGDataset(emg_test, feature_test, label_test, train=False)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 使用改进的模型
        from Models.PyTorchModels_improved_beifen import ImprovedFeaAndEmg_se_PyTorch
        model = ImprovedFeaAndEmg_se_PyTorch(emg_channels, time_steps, feature_dim, num_classes, dropout_rate=dropout_rate).to(device)
        
        # 优化器与LR调度（无需改训练循环）
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        
        # 添加标签平滑的损失函数
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)
        
        best_val_acc = 0.0
        best_model_state = None
        
        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        gate_weights_history = []
        
        early_stopping = EarlyStopping(patience=25, min_delta=0.001)
        
        for epoch in range(epochs):
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            
            for batch_idx, (fea_inputs, emg_inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
                fea_inputs, emg_inputs, labels = fea_inputs.to(device), emg_inputs.to(device), labels.to(device)
                
                # 数据增强
                emg_inputs = emg_augment(emg_inputs)
                
                optimizer.zero_grad()
                outputs, gate_weights = model(fea_inputs, emg_inputs, return_gate_weights=True)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # 每10个batch记录一次门控权重
                if batch_idx % 10 == 0:
                    # 简化门控权重记录，使用detach()分离梯度
                    gate_weights_mean = gate_weights.mean(0).detach().cpu().numpy()
                    gate_weights_history.append((epoch + batch_idx/len(train_loader), gate_weights_mean))

            train_acc = 100.0 * train_correct / train_total
            train_losses.append(train_loss / len(train_loader))
            train_accs.append(train_acc)
            
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for fea_inputs, emg_inputs, labels in val_loader:
                    fea_inputs, emg_inputs, labels = fea_inputs.to(device), emg_inputs.to(device), labels.to(device)
                    
                    # 数据增强
                    emg_inputs = emg_augment(emg_inputs)
                    
                    outputs, _ = model(fea_inputs, emg_inputs, return_gate_weights=True)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = 100.0 * val_correct / val_total
            val_losses.append(val_loss / len(val_loader))
            val_accs.append(val_acc)
            
            scheduler.step()
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.2f}%")
            
            # 监控三件事：Gate均值、参数量、学习率
            with torch.no_grad():
                # 取一小批数据计算Gate统计
                sample_fea, sample_emg, _ = next(iter(train_loader))
                sample_fea, sample_emg = sample_fea[:256].to(device), sample_emg[:256].to(device)
                _, gate_weights = model(sample_fea, sample_emg, return_gate_weights=True)
                gate_mean = gate_weights[:, 0].mean().item()
                gate_std = gate_weights[:, 0].std().item()
                print(f"Gate μ={gate_mean:.3f} σ={gate_std:.3f}")
            
            # 每隔10个epoch统计参数量
            if (epoch + 1) % 10 == 0:
                total_params = sum(p.numel() for p in model.parameters()) / 1e6
                print(f"总参数 {total_params:.2f} M")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
            
            # 早停机制
            if early_stopping(val_losses[-1]):
                print(f"早停机制触发，在第 {epoch+1} 轮停止训练。")
                break

        pltCurve(train_losses, val_losses, train_accs, val_accs, subject_id)
        
        # 绘制门控权重变化
        epochs_hist = [x[0] for x in gate_weights_history]
        # 安全地提取权重，处理可能的维度问题
        space_weights = []
        time_freq_weights = []
        for x in gate_weights_history:
            weights = x[1]
            if isinstance(weights, np.ndarray):
                if weights.size >= 2:
                    space_weights.append(weights[0] * 100)
                    time_freq_weights.append(weights[1] * 100)
                elif weights.size == 1:
                    # 如果只有一个值，假设是空间权重
                    space_weights.append(weights[0] * 100)
                    time_freq_weights.append((1 - weights[0]) * 100)
                else:
                    space_weights.append(50.0)  # 默认值
                    time_freq_weights.append(50.0)
            else:
                space_weights.append(50.0)  # 默认值
                time_freq_weights.append(50.0)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_hist, space_weights, 'b-', label='空间特征权重')
        plt.plot(epochs_hist, time_freq_weights, 'g-', label='时频特征权重')
        plt.title(f'门控权重变化 (被试 {subject_id})')
        plt.xlabel('Epochs')
        plt.ylabel('权重百分比 (%)')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(RESULTS_DIR, f'gate_weights_subject_{subject_id}.png'))
        plt.close()
        # 保存模型权重到results_report目录
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
        
        # 计算并保存当前被试的平均门控权重
        if len(all_gate_weights) > 0:
            # 处理门控权重的不同格式
            processed_weights = []
            for weights in all_gate_weights:
                if isinstance(weights, np.ndarray):
                    if weights.size >= 2:
                        processed_weights.append(weights[:2])  # 只取前两个值
                    elif weights.size == 1:
                        processed_weights.append(np.array([weights[0], 1 - weights[0]]))
                    else:
                        processed_weights.append(np.array([0.5, 0.5]))
                else:
                    processed_weights.append(np.array([0.5, 0.5]))
            
            if processed_weights:
                final_gate_weights = np.mean(processed_weights, axis=0)
            else:
                final_gate_weights = np.array([0.5, 0.5])
        else:
            final_gate_weights = np.array([0.5, 0.5])
        all_subject_gate_weights[subject_id] = final_gate_weights
        
        print(f"受试者 {subject_id} 测试准确率: {test_acc:.2f}%")
        gw = float(final_gate_weights[0]) * 100
        print(f"测试集门控权重：空间特征占 {gw:.2f}% 时频特征占 {100-gw:.2f}%")
        
        cm = confusion_matrix(all_labels, all_preds)
        cm_path = os.path.join(RESULTS_DIR, f'confusion_matrix_subject_{subject_id}.png')
        plot_confusion_matrix(cm, cm_path, f'Subject {subject_id} Confusion Matrix')
        report_str = classification_report(all_labels, all_preds)
        print(report_str)
        # 保存classification_report到txt文件
        report_path = os.path.join(RESULTS_DIR, f'classification_report_subject_{subject_id}.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_str)
        print(f"分类报告已保存到 {report_path}")

    # 打印所有受试者的准确率
    for i, accuracy in enumerate(subject_accuracies):
        print(f"受试者 {i+1} 的准确率: {accuracy:.2f}%")
    
    # 计算总体准确度和统计分析
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
        
        # 计算并打印所有被试的平均门控权重
        avg_gate_weights_all_subjects = np.mean(list(all_subject_gate_weights.values()), axis=0)
        print(f"\n所有被试平均门控权重: 空间特征={avg_gate_weights_all_subjects[0]*100:.2f}%, 时频特征={avg_gate_weights_all_subjects[1]*100:.2f}%")
        
        # 保存结果到文件
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
        
        # 保存到numpy文件
        np.save(os.path.join(RESULTS_DIR, 'subject_accuracies.npy'), results_summary)
        print(f"结果已保存到 {os.path.join(RESULTS_DIR, 'subject_accuracies.npy')}")
    else:
        print("没有收集到任何受试者的准确度数据")

if __name__ == '__main__':
    main() 