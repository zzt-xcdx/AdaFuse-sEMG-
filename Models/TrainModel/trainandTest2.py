"""
====================================================================
清晰版训练脚本 - MS-DSC 消融实验（三组对照）
====================================================================
功能：
  1. 自动循环三个 DTANet 变体（MS-DSC / Single-DSC / Standard）
  2. 为每个变体完整训练所有受试者
  3. 收集精确度、门控权重等统计
  4. 生成对比表格和改进幅度分析
  5. 保存所有结果到独立文件夹

使用方式：
  python trainandtest_ablation_clean.py
====================================================================
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import re
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import h5py
import numpy as np
import matplotlib.pyplot as plt
from Models.PyTorchModels_improved_with_ablation import ImprovedFeaAndEmg_se_PyTorch
from Util.function import get_threeSet
from sklearn.metrics import confusion_matrix, classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam, AdamW
from tqdm import tqdm
import matplotlib
from collections import Counter
import math
import json
import csv

warnings.filterwarnings("ignore", message="h5py is running against HDF5")

# ==================== 配置常量 ====================
DATASET = 'DB4'
LR = 0.0005
DROPOUT_RATE = 0.30
EPOCHS = 150
BATCH_SIZE = 64
EARLY_STOP_PATIENCE = 10
USE_POST_PROCESSING = True

USE_ADAMW = True
WEIGHT_DECAY = 3e-3
LABEL_SMOOTHING = 0.1
USE_CLASS_BALANCED_LOSS = True
CB_BETA = 0.999
GRAD_CLIP_NORM = 1.0

# =============== 数据集配置 ===============
if DATASET == 'DB4':
    root_data = 'D:/DB4'
    subject_range = range(1, 11)
else:  # DB2
    root_data = 'D:/DB2'
    subject_range = range(1, 41)

# =========== 消融实验配置：三种 DTANet 变体 ===========
DTANET_VARIANTS = ["ms_dsc", "single_dsc", "standard"]
BASE_RESULTS_DIR = f'results_ablation_DTANet_{DATASET}'

# 字体设置
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 12

# 随机种子
np.random.seed(123)
torch.manual_seed(123)


# ==================== 辅助工具函数 ====================

def replace_channel_attention_with_identity(model):
    """将模型中的通道注意力模块替换为 nn.Identity()"""
    patterns = [
        re.compile(r'\bSE(Module|Layer|Block)?\b', re.I),
        re.compile(r'SqueezeExcit', re.I),
        re.compile(r'Channel.*Attention', re.I),
        re.compile(r'\bECA(1D)?(Layer|Block)?\b', re.I),
        re.compile(r'Coord.*Atten', re.I),
    ]
    replaced = []

    def _walk(parent, prefix=""):
        for name, child in list(parent.named_children()):
            cls_name = child.__class__.__name__
            full_name = f"{prefix}.{name}" if prefix else name
            if any(p.search(cls_name) for p in patterns) or name.lower() in ("eca", "eca_after"):
                setattr(parent, name, nn.Identity())
                replaced.append((full_name, cls_name))
            else:
                _walk(child, full_name)

    _walk(model, "")

    def _replace_attr_path(obj, attr_path: str):
        parts = attr_path.split(".")
        parent = obj
        for p in parts[:-1]:
            if not hasattr(parent, p):
                return
            parent = getattr(parent, p)
        leaf = parts[-1]
        if hasattr(parent, leaf):
            module = getattr(parent, leaf)
            if not isinstance(module, nn.Identity):
                cls_name = module.__class__.__name__
                setattr(parent, leaf, nn.Identity())
                replaced.append((attr_path, cls_name))

    _replace_attr_path(model, "space_branch.eca")
    _replace_attr_path(model, "time_domain_branch.eca_after")

    return replaced


class EarlyStopping:
    def __init__(self, patience=EARLY_STOP_PATIENCE, min_delta=0.001, restore_best_weights=True):
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
        if len(emg_data.shape) == 4:
            emg_np = emg_data[..., 0]
        elif len(emg_data.shape) == 3:
            if emg_data.shape[-1] in (12, 16):
                emg_np = np.transpose(emg_data, (0, 2, 1))
            else:
                emg_np = emg_data
        else:
            raise ValueError(f"不支持的 EMG 数据维度: {emg_data.shape}")

        mean = emg_np.mean(axis=2, keepdims=True)
        std = emg_np.std(axis=2, keepdims=True) + 1e-6
        emg_np = (emg_np - mean) / std
        emg_np = emg_np[..., np.newaxis]

        self.emg_data = torch.FloatTensor(emg_np)
        self.feature_data = torch.FloatTensor(feature_data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.emg_data[idx], self.feature_data[idx], self.labels[idx]


def compute_class_balanced_weights(labels_np, num_classes, beta=0.999):
    """基于有效样本数计算类别权重"""
    counts = np.bincount(labels_np, minlength=num_classes).astype(np.int64)
    nz = counts[counts > 0]
    if len(nz) == 0:
        return torch.ones(num_classes, dtype=torch.float32), counts
    min_nz = nz.min()
    counts[counts == 0] = min_nz
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
    weights = weights * (num_classes / np.sum(weights))
    return torch.tensor(weights, dtype=torch.float32), counts


class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing + 可选类别权重"""

    def __init__(self, num_classes, smoothing=0.0, weight=None):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.register_buffer('weight', weight if weight is not None else None)

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / self.num_classes)
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        loss_per_sample = (-true_dist * log_probs).sum(dim=1)
        if self.weight is not None:
            w = self.weight.gather(0, target)
            loss_per_sample = loss_per_sample * w
        return loss_per_sample.mean()


def build_ce_loss(num_classes, class_weights_t, label_smoothing, device):
    """构建交叉熵损失"""
    if label_smoothing > 0:
        try:
            return nn.CrossEntropyLoss(
                weight=(class_weights_t.to(device) if class_weights_t is not None else None),
                label_smoothing=label_smoothing
            ).to(device)
        except TypeError:
            return LabelSmoothingCrossEntropy(
                num_classes=num_classes,
                smoothing=label_smoothing,
                weight=(class_weights_t.to(device) if class_weights_t is not None else None)
            ).to(device)
    else:
        return nn.CrossEntropyLoss(
            weight=(class_weights_t.to(device) if class_weights_t is not None else None)
        ).to(device)


def majority_vote_probs(logits: torch.Tensor, k: int = 7) -> np.ndarray:
    """多数投票平滑"""
    assert k % 2 == 1, "k 必须是奇数"
    assert logits.dim() == 2, "logits 形状应为 [T, C]"
    T, C = logits.shape
    r = k // 2
    pad = F.pad(logits.T.unsqueeze(0), (r, r, 0, 0), mode="replicate").squeeze(0).T
    out = []
    for t in range(T):
        win_logits = pad[t:t + k]
        probs = F.softmax(win_logits, dim=-1)
        avg_prob = probs.mean(dim=0)
        out.append(int(torch.argmax(avg_prob)))
    return np.array(out, dtype=int)


def plot_training_curves(subject_id, train_losses, val_losses, train_accs, val_accs, save_dir):
    """绘制训练曲线"""
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', linewidth=2, label='训练损失')
    plt.plot(val_losses, 'r-', linewidth=2, label='验证损失')
    plt.title(f'损失曲线 (被试 {subject_id})', fontsize=14, fontweight='bold')
    plt.xlabel('轮次', fontsize=12)
    plt.ylabel('损失值', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, 'b-', linewidth=2, label='训练准确率')
    plt.plot(val_accs, 'r-', linewidth=2, label='验证准确率')
    plt.title(f'准确率曲线 (被试 {subject_id})', fontsize=14, fontweight='bold')
    plt.xlabel('轮次', fontsize=12)
    plt.ylabel('准确率 (%)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(save_dir, f'training_curves_s{subject_id}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path


def plot_confusion_matrix(cm, subject_id, save_path):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, aspect='auto')
    plt.title(f'混淆矩阵 - 被试 {subject_id}', fontsize=16, fontweight='bold')
    cbar = plt.colorbar()
    cbar.set_label('样本数', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    thresh = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ==================== 主训练函数 ====================

def train_single_subject(subject_id, variant, device, variant_results_dir):
    """为单个受试者训练一个 DTANet 变体"""
    try:
        emg_file = os.path.join(root_data, f'data/restimulus/reSegEnhance/{DATASET}_s{subject_id}SegEnhance.h5')
        feature_file = os.path.join(root_data, f'data/restimulus/Fea/{DATASET}_s{subject_id}feaEnhance.h5')

        if not os.path.exists(emg_file) or not os.path.exists(feature_file):
            print(f"  ✗ 被试 {subject_id} 数据文件缺失")
            return None

        # 加载数据
        with h5py.File(emg_file, 'r') as f:
            emg_data = f['emg'][:]
            labels = f['label'][:]
            rep_arr = f['rep'][:]
        with h5py.File(feature_file, 'r') as f:
            feature_data = f['features'][:]

        # 数据划分
        rep_vali = 2
        emg_train, emg_vali, emg_test, label_train, label_vali, label_test = \
            get_threeSet(emg_data, labels, rep_arr, rep_vali)
        feature_train, feature_vali, feature_test, _, _, _ = \
            get_threeSet(feature_data, labels, rep_arr, rep_vali)

        # 构建数据加载器
        train_dataset = EMGDataset(emg_train, feature_train, label_train)
        val_dataset = EMGDataset(emg_vali, feature_vali, label_vali)
        test_dataset = EMGDataset(emg_test, feature_test, label_test)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # 确定维度
        if emg_train.ndim == 3 and emg_train.shape[-1] in (12, 16):
            emg_channels = int(emg_train.shape[-1])
            time_steps = int(emg_train.shape[1])
        else:
            emg_channels = int(emg_train.shape[1])
            time_steps = int(emg_train.shape[2])

        num_classes = int(np.max(label_train) + 1)
        feature_dim = int(feature_train.shape[1])

        # 构建模型【关键：传入 dtanet_variant 参数】
        model = ImprovedFeaAndEmg_se_PyTorch(
            emg_channels=emg_channels,
            time_steps=time_steps,
            feature_dim=feature_dim,
            num_classes=num_classes,
            dropout_rate=DROPOUT_RATE,
            dtanet_variant=variant  # ← 三种变体在这里切换
        ).to(device)

        # 移除通道注意力
        replaced = replace_channel_attention_with_identity(model)
        if replaced:
            print(f"    已移除 {len(replaced)} 个通道注意力模块")

        # 损失函数
        class_weights_t = None
        if USE_CLASS_BALANCED_LOSS:
            class_weights_t, _ = compute_class_balanced_weights(label_train, num_classes, beta=CB_BETA)

        criterion = build_ce_loss(num_classes, class_weights_t, LABEL_SMOOTHING, device)

        # 优化器
        if USE_ADAMW:
            optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        else:
            optimizer = Adam(model.parameters(), lr=LR)

        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8, min_lr=1e-5, verbose=False)
        early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE, min_delta=0.001)

        # 训练循环
        best_val_acc = 0.0
        best_model_state = None
        train_losses, val_losses, train_accs, val_accs = [], [], [], []

        for epoch in range(EPOCHS):
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            for emg_inputs, fea_inputs, labels in train_loader:
                emg_inputs = emg_inputs.to(device)
                fea_inputs = fea_inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(fea_inputs, emg_inputs, return_gate_weights=False)
                loss = criterion(outputs, labels)
                loss.backward()

                if GRAD_CLIP_NORM is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            train_acc = 100.0 * train_correct / max(1, train_total)
            train_losses.append(train_loss / max(1, len(train_loader)))
            train_accs.append(train_acc)

            # 验证
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for emg_inputs, fea_inputs, labels in val_loader:
                    emg_inputs = emg_inputs.to(device)
                    fea_inputs = fea_inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(fea_inputs, emg_inputs, return_gate_weights=False)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_acc = 100.0 * val_correct / max(1, val_total)
            val_losses.append(val_loss / max(1, len(val_loader)))
            val_accs.append(val_acc)

            scheduler.step(val_acc)

            if epoch % 10 == 0:
                print(f"      Epoch {epoch + 1:3d}: Train={train_acc:6.2f}% | Val={val_acc:6.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()

            if early_stopping(val_losses[-1], model):
                print(f"      早停于第 {epoch + 1} 轮")
                break

        # 绘制曲线
        plot_training_curves(subject_id, train_losses, val_losses, train_accs, val_accs, variant_results_dir)

        # 测试
        if best_model_state is None:
            best_model_state = model.state_dict().copy()
        model.load_state_dict(best_model_state)
        model.eval()

        all_preds, all_labels, logits_list = [], [], []
        with torch.no_grad():
            for emg_inputs, fea_inputs, labels in test_loader:
                emg_inputs = emg_inputs.to(device)
                fea_inputs = fea_inputs.to(device)
                labels = labels.to(device)
                outputs = model(fea_inputs, emg_inputs, return_gate_weights=False)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                logits_list.append(outputs.detach().cpu())

        all_logits = torch.cat(logits_list, dim=0)
        all_labels_np = np.array(all_labels)
        all_preds_np = np.array(all_preds)

        # 后处理
        if USE_POST_PROCESSING:
            preds_final = majority_vote_probs(all_logits, k=7)
        else:
            preds_final = all_preds_np

        test_acc = 100.0 * np.sum(preds_final == all_labels_np) / len(all_labels_np)

        # 混淆矩阵
        cm = confusion_matrix(all_labels_np, preds_final)
        cm_path = os.path.join(variant_results_dir, f'cm_s{subject_id}.png')
        plot_confusion_matrix(cm, subject_id, cm_path)

        print(f"    ✓ 被试 {subject_id} 完成 | 测试准确率: {test_acc:.2f}%")
        return test_acc

    except Exception as e:
        print(f"  ✗ 被试 {subject_id} 出错: {e}")
        return None


def train_variant(variant, device):
    """为一个 DTANet 变体训练所有受试者"""
    variant_dir = os.path.join(BASE_RESULTS_DIR, f'variant_{variant}')
    os.makedirs(variant_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"开始训练变体: {variant.upper()}")
    print(f"结果保存路径: {variant_dir}")
    print(f"{'=' * 70}")

    subject_accs = []
    for subject_id in subject_range:
        print(f"\n[{variant}] 处理被试 {subject_id}...")
        acc = train_single_subject(subject_id, variant, device, variant_dir)
        if acc is not None:
            subject_accs.append(acc)

    if len(subject_accs) == 0:
        print(f"✗ 变体 {variant} 没有成功处理任何受试者")
        return None

    results = {
        'variant': variant,
        'oa': float(np.mean(subject_accs)),
        'std': float(np.std(subject_accs)),
        'max': float(np.max(subject_accs)),
        'min': float(np.min(subject_accs)),
        'num_subjects': len(subject_accs),
        'subject_accuracies': [float(x) for x in subject_accs]
    }

    # 保存结果
    results_json_path = os.path.join(variant_dir, 'results.json')
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[{variant}] 汇总结果:")
    print(f"  总体准确度 (OA): {results['oa']:.2f}%")
    print(f"  准确度标准差: {results['std']:.2f}%")
    print(f"  最高准确度: {results['max']:.2f}%")
    print(f"  最低准确度: {results['min']:.2f}%")
    print(f"  处理受试者数: {results['num_subjects']}")

    return results


def generate_comparison_report(all_results):
    """生成对比报告"""
    print(f"\n{'=' * 70}")
    print("三组对照实验 - 对比总结")
    print(f"{'=' * 70}\n")

    # 表格头
    print(f"{'变体':<15} {'OA':<12} {'std':<12} {'max':<12} {'min':<12} {'样本数':<10}")
    print("-" * 70)

    for result in all_results:
        variant = result['variant'].upper()
        oa = result['oa']
        std = result['std']
        max_acc = result['max']
        min_acc = result['min']
        num_subj = result['num_subjects']
        print(f"{variant:<15} {oa:>10.2f}% {std:>10.2f}% {max_acc:>10.2f}% {min_acc:>10.2f}% {num_subj:>8}")

    # 计算改进幅度
    print("\n改进幅度分析:")
    print("-" * 70)

    ms_dsc_result = next((r for r in all_results if r['variant'] == 'ms_dsc'), None)
    single_dsc_result = next((r for r in all_results if r['variant'] == 'single_dsc'), None)
    standard_result = next((r for r in all_results if r['variant'] == 'standard'), None)

    if ms_dsc_result and standard_result:
        improvement_vs_std = ms_dsc_result['oa'] - standard_result['oa']
        print(
            f"MS-DSC vs Standard Conv: {improvement_vs_std:+.2f}% (MS-DSC 更优)" if improvement_vs_std > 0 else f"MS-DSC vs Standard Conv: {improvement_vs_std:+.2f}% (Standard 更优)")

    if ms_dsc_result and single_dsc_result:
        improvement_vs_single = ms_dsc_result['oa'] - single_dsc_result['oa']
        print(
            f"MS-DSC vs Single-DSC: {improvement_vs_single:+.2f}% (MS-DSC 更优)" if improvement_vs_single > 0 else f"MS-DSC vs Single-DSC: {improvement_vs_single:+.2f}% (Single-DSC 更优)")

    if single_dsc_result and standard_result:
        improvement_dsc_vs_std = single_dsc_result['oa'] - standard_result['oa']
        print(
            f"Single-DSC vs Standard Conv: {improvement_dsc_vs_std:+.2f}% (Single-DSC 更优)" if improvement_dsc_vs_std > 0 else f"Single-DSC vs Standard Conv: {improvement_dsc_vs_std:+.2f}% (Standard 更优)")

    # 保存对比表
    comparison_csv = os.path.join(BASE_RESULTS_DIR, 'comparison_table.csv')
    with open(comparison_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['变体', 'OA (%)', 'std (%)', 'max (%)', 'min (%)', '样本数'])
        for result in all_results:
            writer.writerow([
                result['variant'],
                f"{result['oa']:.2f}",
                f"{result['std']:.2f}",
                f"{result['max']:.2f}",
                f"{result['min']:.2f}",
                result['num_subjects']
            ])

    # 保存完整结果
    comparison_json = os.path.join(BASE_RESULTS_DIR, 'comparison_results.json')
    with open(comparison_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 对比表已保存至: {comparison_csv}")
    print(f"✓ 完整结果已保存至: {comparison_json}")


# ==================== 主程序入口 ====================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    print(f"数据集: {DATASET}")
    print(f"总轮次: {EPOCHS} | 批大小: {BATCH_SIZE}")

    os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

    all_results = []

    # 循环三个变体
    for variant in DTANET_VARIANTS:
        result = train_variant(variant, device)
        if result:
            all_results.append(result)

    # 生成对比报告
    if len(all_results) > 0:
        generate_comparison_report(all_results)
        print(f"\n✓ 所有实验完成！结果保存至: {BASE_RESULTS_DIR}")
    else:
        print("\n✗ 没有成功完成任何实验")


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception as e:
        print(f"\n✗ 程序异常: {e}")
        traceback.print_exc()
