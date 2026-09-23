#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DB5数据集训练脚本 - 使用预处理的Stockwell时频图
直接加载预处理的时频图，避免训练时动态计算
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PyTorchModels_improved_DB5 import ImprovedFeaAndEmg_se_PyTorch_DB5
from Util.function import get_threeSet_indices

class EMGStockwellDataset(Dataset):
    """
    EMG数据集 - 加载原始EMG和预处理的Stockwell时频图
    """
    def __init__(self, emg_file, stockwell_file, indices=None):
        """
        Args:
            emg_file: 原始EMG数据文件路径
            stockwell_file: 预处理的Stockwell时频图文件路径
            indices: 要使用的数据索引，如果为None则使用全部
        """
        print(f"加载EMG数据: {emg_file}")
        with h5py.File(emg_file, 'r') as f:
            self.emg = f['emg'][:]  # (N, 400, 16)
            self.label = f['label'][:] - 1  # 标签从0开始
            self.rep = f['rep'][:]
        
        print(f"加载Stockwell时频图: {stockwell_file}")
        with h5py.File(stockwell_file, 'r') as f:
            self.stockwell = f['stockwell'][:]  # (N, 16, 32, 400)
        
        # 如果指定了索引，则只使用部分数据
        if indices is not None:
            self.emg = self.emg[indices]
            self.stockwell = self.stockwell[indices]
            self.label = self.label[indices]
            self.rep = self.rep[indices]
        
        print(f"数据集大小: {len(self.label)}")
        print(f"EMG形状: {self.emg.shape}")
        print(f"Stockwell形状: {self.stockwell.shape}")
        print(f"标签范围: {self.label.min()} - {self.label.max()}")
        print(f"重复次数: {np.unique(self.rep)}")
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        # 转换为torch tensor
        emg = torch.from_numpy(self.emg[idx]).float()  # (400, 16)
        stockwell = torch.from_numpy(self.stockwell[idx]).float()  # (16, 32, 400)
        label = int(self.label[idx])
        
        return emg, stockwell, label

def train_model(model, train_loader, val_loader, device, num_epochs=100, lr=0.001):
    """
    训练模型
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    
    # 记录训练过程
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0.0
    patience_counter = 0
    patience = 20
    
    print(f"开始训练，设备: {device}")
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (emg, stockwell, labels) in enumerate(train_pbar):
            emg, stockwell, labels = emg.to(device), stockwell.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs, gate_weights = model(emg, stockwell)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for emg, stockwell, labels in val_pbar:
                emg, stockwell, labels = emg.to(device), stockwell.to(device), labels.to(device)
                
                outputs, gate_weights = model(emg, stockwell)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 学习率调度
        scheduler.step(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), f'best_model_subject_{subject_id}.pth')
            print(f'  保存最佳模型，验证准确率: {best_val_acc:.2f}%')
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= patience:
            print(f'早停触发，{patience}个epoch没有改善')
            break
    
    return train_losses, train_accs, val_losses, val_accs, best_val_acc

def evaluate_model(model, test_loader, device, subject_id):
    """
    评估模型
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    print("开始测试...")
    with torch.no_grad():
        for emg, stockwell, labels in tqdm(test_loader, desc="测试"):
            emg, stockwell, labels = emg.to(device), stockwell.to(device), labels.to(device)
            
            outputs, gate_weights = model(emg, stockwell)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算准确率
    accuracy = 100. * sum(1 for x, y in zip(all_predictions, all_labels) if x == y) / len(all_labels)
    print(f"测试准确率: {accuracy:.2f}%")
    
    # 生成分类报告
    report = classification_report(all_labels, all_predictions, digits=4)
    print("分类报告:")
    print(report)
    
    # 保存分类报告
    with open(f'classification_report_subject_{subject_id}.txt', 'w', encoding='utf-8') as f:
        f.write(f"被试 {subject_id} 分类报告\n")
        f.write(f"测试准确率: {accuracy:.2f}%\n\n")
        f.write(report)
    
    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'被试 {subject_id} 混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_subject_{subject_id}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return accuracy, report

def plot_training_curves(train_losses, train_accs, val_losses, val_accs, subject_id):
    """
    绘制训练曲线
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    ax1.plot(train_losses, label='训练损失', color='blue')
    ax1.plot(val_losses, label='验证损失', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('训练和验证损失')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(train_accs, label='训练准确率', color='blue')
    ax2.plot(val_accs, label='验证准确率', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('训练和验证准确率')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'training_curves_subject_{subject_id}.png', dpi=300, bbox_inches='tight')
    plt.close()

def train_single_subject(subject_id, batch_size=32, num_epochs=100, learning_rate=0.001):
    """
    训练单个被试
    """
    print(f"\n{'='*60}")
    print(f"开始训练被试 {subject_id}")
    print(f"{'='*60}")
    
    # 文件路径
    emg_file = f"D:/DB5/data/restimulus/reSegEnhance/DB5_s{subject_id}SegEnhance.h5"
    stockwell_file = f"D:/DB5/data/restimulus/Stockwell/DB5_s{subject_id}_stockwell.h5"
    
    # 检查文件是否存在
    if not os.path.exists(emg_file):
        print(f"错误: EMG文件不存在 {emg_file}")
        return None
    
    if not os.path.exists(stockwell_file):
        print(f"错误: Stockwell文件不存在 {stockwell_file}")
        print("请先运行 preprocess_stockwell_DB5.py 生成时频图")
        return None
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    try:
        # 加载数据
        print("加载数据集...")
        full_dataset = EMGStockwellDataset(emg_file, stockwell_file)
        
        # 数据分割
        print("分割数据集...")
        # 使用重复次数2作为验证集
        train_indices, val_indices, test_indices = get_threeSet_indices(
            full_dataset.rep, 
            2  # 使用重复次数2作为验证集
        )
        
        # 创建数据加载器
        train_dataset = EMGStockwellDataset(emg_file, stockwell_file, train_indices)
        val_dataset = EMGStockwellDataset(emg_file, stockwell_file, val_indices)
        test_dataset = EMGStockwellDataset(emg_file, stockwell_file, test_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # 创建模型
        print("创建模型...")
        model = ImprovedFeaAndEmg_se_PyTorch_DB5(num_classes=52).to(device)
        
        # 打印模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        
        # 训练模型
        print("开始训练...")
        train_losses, train_accs, val_losses, val_accs, best_val_acc = train_model(
            model, train_loader, val_loader, device, num_epochs, learning_rate
        )
        
        # 加载最佳模型
        model.load_state_dict(torch.load(f'best_model_subject_{subject_id}.pth'))
        
        # 评估模型
        print("评估模型...")
        test_accuracy, report = evaluate_model(model, test_loader, device, subject_id)
        
        # 绘制训练曲线
        print("绘制训练曲线...")
        plot_training_curves(train_losses, train_accs, val_losses, val_accs, subject_id)
        
        # 保存结果
        results = {
            'subject_id': subject_id,
            'best_val_acc': best_val_acc,
            'test_accuracy': test_accuracy,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs
        }
        
        print(f"\n{'='*60}")
        print(f"被试 {subject_id} 训练完成！")
        print(f"最佳验证准确率: {best_val_acc:.2f}%")
        print(f"测试准确率: {test_accuracy:.2f}%")
        print(f"{'='*60}")
        
        return results
        
    except Exception as e:
        print(f"训练被试 {subject_id} 时出错: {str(e)}")
        return None

def main():
    """
    主函数：批量训练所有被试
    """
    # 配置参数
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.001
    
    # 被试列表 (DB5有10个被试)
    subjects = list(range(1, 11))
    
    print("DB5数据集批量训练")
    print(f"被试列表: {subjects}")
    print(f"批次大小: {batch_size}")
    print(f"训练轮数: {num_epochs}")
    print(f"学习率: {learning_rate}")
    
    # 创建结果目录
    results_dir = "results_report_DB5_stockwell"
    os.makedirs(results_dir, exist_ok=True)
    
    # 批量训练
    all_results = []
    success_count = 0
    
    for subject_id in subjects:
        # 切换到结果目录
        os.chdir(results_dir)
        
        # 训练单个被试
        result = train_single_subject(subject_id, batch_size, num_epochs, learning_rate)
        
        if result is not None:
            all_results.append(result)
            success_count += 1
        
        # 返回原目录
        os.chdir("..")
    
    # 汇总结果
    print(f"\n{'='*80}")
    print("批量训练完成！")
    print(f"成功训练: {success_count}/{len(subjects)} 个被试")
    print(f"{'='*80}")
    
    if all_results:
        # 计算平均准确率
        val_accs = [r['best_val_acc'] for r in all_results]
        test_accs = [r['test_accuracy'] for r in all_results]
        
        print(f"平均验证准确率: {np.mean(val_accs):.2f}% ± {np.std(val_accs):.2f}%")
        print(f"平均测试准确率: {np.mean(test_accs):.2f}% ± {np.std(test_accs):.2f}%")
        
        # 保存汇总结果
        summary_file = os.path.join(results_dir, "training_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("DB5数据集批量训练汇总\n")
            f.write("="*50 + "\n")
            f.write(f"成功训练被试数: {success_count}/{len(subjects)}\n")
            f.write(f"平均验证准确率: {np.mean(val_accs):.2f}% ± {np.std(val_accs):.2f}%\n")
            f.write(f"平均测试准确率: {np.mean(test_accs):.2f}% ± {np.std(test_accs):.2f}%\n\n")
            
            f.write("各被试详细结果:\n")
            f.write("-"*30 + "\n")
            for result in all_results:
                f.write(f"被试 {result['subject_id']}:\n")
                f.write(f"  验证准确率: {result['best_val_acc']:.2f}%\n")
                f.write(f"  测试准确率: {result['test_accuracy']:.2f}%\n")
                f.write(f"  模型参数: {result['total_params']:,}\n\n")
        
        # 绘制所有被试的准确率对比图
        plt.figure(figsize=(12, 6))
        x = [r['subject_id'] for r in all_results]
        plt.plot(x, val_accs, 'o-', label='验证准确率', color='blue')
        plt.plot(x, test_accs, 's-', label='测试准确率', color='red')
        plt.xlabel('被试编号')
        plt.ylabel('准确率 (%)')
        plt.title('DB5数据集各被试准确率对比')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'all_subjects_accuracy.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"汇总结果已保存到: {summary_file}")
        print(f"准确率对比图已保存到: {os.path.join(results_dir, 'all_subjects_accuracy.png')}")

if __name__ == "__main__":
    main() 