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
from Models.PyTorchModels_improved_with_ablation import FeaAndEmg_se_PyTorch
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

# 数据配置
root_data = 'D:/DB4'
subject_range = [1]  # 只测试被试1

# 结果保存目录
RESULTS_DIR = 'results_discrete_stockwell'
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
    
    if subject_id is not None:
        filename = os.path.join(RESULTS_DIR, f'training_curves_discrete_stockwell_subject_{subject_id}.png')
    else:
        filename = os.path.join(RESULTS_DIR, 'training_curves_discrete_stockwell.png')
    
    plt.savefig(filename)
    plt.close()
    print(f"训练曲线已保存到 {filename}")

class EMGDataset(Dataset):
    def __init__(self, emg_data, feature_data, labels):
        if len(emg_data.shape) == 4:
            self.emg_data = torch.FloatTensor(emg_data)
        else:
            raise ValueError(f"不支持的EMG数据维度: {emg_data.shape}")
        self.feature_data = torch.FloatTensor(feature_data)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.feature_data[idx], self.emg_data[idx], self.labels[idx]

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
    cbar = plt.colorbar()
    cbar.set_label('样本数量', fontsize=12)
    plt.savefig(savename, format='png')
    plt.close()

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置超参数
    batch_size = 32
    dropout_rate = 0.4
    epochs = 100
    
    # 遍历被试
    for subject_id in subject_range:
        print(f"\n=== 处理被试 {subject_id} - 使用离散Stockwell特征 ===")
        
        # 数据文件路径
        emg_file = os.path.join(root_data, f'data/restimulus/reSegNoEnhance/DB4_s{subject_id}SegNoEnhance.h5')
        feature_file = os.path.join(root_data, f'data/restimulus/FeaDiscreteStockwell/DB4_s{subject_id}feaDiscreteStockwell.h5')
        
        # 检查文件是否存在
        if not os.path.exists(emg_file):
            print(f"错误: EMG文件未找到: {emg_file}")
            continue
        if not os.path.exists(feature_file):
            print(f"错误: 离散Stockwell特征文件未找到: {feature_file}")
            print("请先运行 feature/GetFeature_discrete_stockwell.py 生成特征")
            continue
        
        try:
            # 加载数据
            print("加载数据...")
            with h5py.File(emg_file, 'r') as f:
                emg_data = f['emg'][:]
                labels = f['label'][:]
                rep_arr = f['rep'][:]
            
            with h5py.File(feature_file, 'r') as f:
                feature_data = f['features'][:]
            
            print(f"EMG数据形状: {emg_data.shape}")
            print(f"离散Stockwell特征形状: {feature_data.shape}")
            
            # 使用get_threeSet按重复次数划分数据集
            rep_vali = 2
            emg_train, emg_vali, emg_test, label_train, label_vali, label_test = get_threeSet(emg_data, labels, rep_arr, rep_vali)
            feature_train, feature_vali, feature_test, _, _, _ = get_threeSet(feature_data, labels, rep_arr, rep_vali)

            # 数据集划分验证
            train_indices = np.where(np.isin(rep_arr, [1, 3, 4, 6]))[0]
            val_indices = np.where(rep_arr == 2)[0]
            test_indices = np.where(rep_arr == 5)[0]
            
            print(f"训练集: {len(train_indices)} 样本")
            print(f"验证集: {len(val_indices)} 样本")
            print(f"测试集: {len(test_indices)} 样本")
            
            # 标签映射
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
            
            print(f"模型参数: channels={emg_channels}, time_steps={time_steps}, feature_dim={feature_dim}, classes={num_classes}")
            
            # 重塑EMG数据
            emg_train = emg_train.reshape(emg_train.shape[0], emg_train.shape[1], emg_train.shape[2], 1)
            emg_vali = emg_vali.reshape(emg_vali.shape[0], emg_vali.shape[1], emg_vali.shape[2], 1)
            emg_test = emg_test.reshape(emg_test.shape[0], emg_test.shape[1], emg_test.shape[2], 1)
            
        except Exception as e:
            print(f"加载数据时出错: {e}")
            continue

        # 创建数据加载器
        train_dataset = EMGDataset(emg_train, feature_train, label_train)
        val_dataset = EMGDataset(emg_vali, feature_vali, label_vali)
        test_dataset = EMGDataset(emg_test, feature_test, label_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 创建模型 (使用正确的参数顺序)
        model = FeaAndEmg_se_PyTorch(emg_channels, time_steps, feature_dim, num_classes, dropout_rate=dropout_rate).to(device)
        
        # 计算模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型总参数量: {total_params:,}")
        
        optimizer = NAdam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss().to(device)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
        
        best_val_acc = 0.0
        best_model_state = None
        
        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        gate_weights_history = []
        
        print("\n开始训练...")
        for epoch in range(epochs):
            # 训练阶段
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
                
                # 记录门控权重
                if batch_idx % 10 == 0:
                    gate_weights_mean = gate_weights.mean(dim=0).detach().cpu().numpy()
                    gate_weights_history.append((epoch + batch_idx/len(train_loader), gate_weights_mean))

            train_acc = 100.0 * train_correct / train_total
            train_losses.append(train_loss / len(train_loader))
            train_accs.append(train_acc)
            
            # 验证阶段
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

        # 保存训练曲线
        pltCurve(train_losses, val_losses, train_accs, val_accs, subject_id)
        
        # 绘制门控权重变化
        epochs_hist = [x[0] for x in gate_weights_history]
        space_weights = [x[1][0] * 100 for x in gate_weights_history]
        time_freq_weights = [x[1][1] * 100 for x in gate_weights_history]
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_hist, space_weights, 'b-', label='空间特征权重')
        plt.plot(epochs_hist, time_freq_weights, 'g-', label='时频特征权重')
        plt.title(f'门控权重变化 (被试 {subject_id} - 离散Stockwell)')
        plt.xlabel('Epochs')
        plt.ylabel('权重百分比 (%)')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(RESULTS_DIR, f'gate_weights_discrete_stockwell_subject_{subject_id}.png'))
        plt.close()
        
        # 保存最佳模型
        torch.save(best_model_state, os.path.join(RESULTS_DIR, f'best_model_discrete_stockwell_subject_{subject_id}.pth'))
        print(f"最佳模型已保存")

        # 测试阶段
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
        final_gate_weights = np.mean(np.concatenate(all_gate_weights, axis=0), axis=0)
        
        print(f"\n=== 测试结果 (离散Stockwell) ===")
        print(f"被试 {subject_id} 测试准确率: {test_acc:.2f}%")
        print(f"门控权重: 空间特征={final_gate_weights[0]*100:.2f}%, 时频特征={final_gate_weights[1]*100:.2f}%")
        
        # 保存混淆矩阵和分类报告
        cm = confusion_matrix(all_labels, all_preds)
        cm_path = os.path.join(RESULTS_DIR, f'confusion_matrix_discrete_stockwell_subject_{subject_id}.png')
        plot_confusion_matrix(cm, cm_path, f'Subject {subject_id} Confusion Matrix (Discrete Stockwell)')
        
        report_str = classification_report(all_labels, all_preds)
        report_path = os.path.join(RESULTS_DIR, f'classification_report_discrete_stockwell_subject_{subject_id}.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"被试 {subject_id} 分类报告 (离散Stockwell特征)\n")
            f.write(f"测试准确率: {test_acc:.2f}%\n")
            f.write(f"门控权重: 空间={final_gate_weights[0]*100:.2f}%, 时频={final_gate_weights[1]*100:.2f}%\n\n")
            f.write(report_str)
        
        print(f"结果已保存到 {RESULTS_DIR}/")

if __name__ == '__main__':
    main() 