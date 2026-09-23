
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
from Models.PyTorchModels_improved_beifen import ImprovedFeaAndEmg_V2
from Util.function import get_threeSet
from sklearn.metrics import confusion_matrix, classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam, AdamW
from tqdm import tqdm
import matplotlib
from collections import Counter
from collections import defaultdict
import math


#
warnings.filterwarnings("ignore", message="h5py is running against HDF5")

# ---------------- Config 开关（可按需改） ----------------
DATASET = 'DB4'          # 'DB2' or 'DB4'
LR = 0.0005
DROPOUT_RATE = 0.30 #原来0.3
EPOCHS = 150
BATCH_SIZE = 64
EARLY_STOP_PATIENCE = 10
USE_POST_PROCESSING = True  #当前使用多数投票后处理

# 消融实验开关：可选 None / 'space' / 'time_dom' / 'time_freq'
ABLATED_BRANCH = 'None'
# 也可以用环境变量切换（可选）：ABLATED_BRANCH = (os.environ.get('ABLATE_BRANCH') or '').strip() or None


# 训练增强
USE_ADAMW = True
WEIGHT_DECAY = 3e-3
LABEL_SMOOTHING = 0.1   # 设为 0 可关闭
USE_CLASS_BALANCED_LOSS = True
CB_BETA = 0.999          # 有效样本数权重的 beta
GRAD_CLIP_NORM = 1.0     # 设为 None 关闭
WARMUP_EPOCHS = 5
MIN_LR_RATIO = 0.1  # 余弦末端最低 LR = base_lr * 0.1
# -------------------------------------------------------

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

# 数据集路径与被试范围
if DATASET == 'DB4':
    root_data = 'D:/DB4'
    subject_range = range(1, 11)  # 只训练受试者7和8
else:  # DB2
    root_data = 'D:/DB2'
    subject_range = range(1, 41)  # DB2有40个被试

# 结果保存目录
RESULTS_DIR = f'results_report_finegate_{DATASET}_minimal_rollback'
os.makedirs(RESULTS_DIR, exist_ok=True)


def replace_channel_attention_with_identity(model):
    """
    将模型中的“通道注意力模块”替换为 nn.Identity()。
    兼容类名（忽略大小写）：
      - SE/SEModule/SELayer/SEBlock/SqueezeExcite/SqueezeExcitation
      - ChannelAttention/CBAMChannelAttention
      - ECA/ECA1D/ECALayer/ECABlock
      - CoordAtt/CoordAttention（若含通道重标定也会替换）
    同时包含已知属性名兜底：space_branch.eca、time_domain_branch.eca_after
    返回：[(模块路径, 原类名), ...]
    """
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
            if any(p.search(cls_name) for p in patterns) or name.lower() in ("eca", "eca_after", "se", "semodule", "se_layer", "channel_attention"):
                setattr(parent, name, nn.Identity())
                replaced.append((full_name, cls_name))
            else:
                _walk(child, full_name)

    _walk(model, "")

    # 已知路径兜底
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
        # 支持 3 维 (N,T,C)/(N,C,T) 或 4 维 (N,C,T,1)
        if len(emg_data.shape) == 4:
            emg_np = emg_data[..., 0]  # (N, C, T, 1) → (N, C, T)
        elif len(emg_data.shape) == 3:
            if emg_data.shape[-1] in (12, 16):           # (N, T, C)
                emg_np = np.transpose(emg_data, (0, 2, 1))  # (N, C, T)
            else:                                         # (N, C, T)
                emg_np = emg_data
        else:
            raise ValueError(f"不支持的 EMG 数据维度: {emg_data.shape}")

        # 逐窗口 z-score（沿时间轴 T）
        mean = emg_np.mean(axis=2, keepdims=True)              # (N, C, 1)
        std  = emg_np.std(axis=2, keepdims=True) + 1e-6
        emg_np = (emg_np - mean) / std                         # (N, C, T)

        # 加回最后 1 维，变成 (N, C, T, 1)
        emg_np = emg_np[..., np.newaxis]

        # 转张量
        self.emg_data = torch.FloatTensor(emg_np)
        self.feature_data = torch.FloatTensor(feature_data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 返回顺序：emg, feature, label
        return self.emg_data[idx], self.feature_data[idx], self.labels[idx]


def majority_filter(seq, k=7):
    """对序列进行多数投票平滑（当前未启用，仅保留函数）"""
    assert k % 2 == 1
    r = k // 2
    out = list(seq)
    for i in range(len(seq)):
        s = seq[max(0, i-r):min(len(seq), i+r+1)]
        out[i] = Counter(s).most_common(1)[0][0]
    return np.array(out)

def build_warmup_cosine(optimizer, total_epochs, warmup_epochs=5, min_lr_ratio=0.1):
    """
    前 warmup_epochs 线性升温到 base LR；之后余弦退火到 min_lr_ratio * base LR
    返回：torch.optim.lr_scheduler.LambdaLR（按 epoch 调度）
    """
    warmup_epochs = max(0, int(warmup_epochs))
    min_lr_ratio = float(min_lr_ratio)

    def lr_lambda(epoch):
        # epoch 从 0 开始
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        rem = max(1, total_epochs - warmup_epochs)
        t = min(1.0, max(0.0, (epoch - warmup_epochs) / rem))
        cos = 0.5 * (1.0 + math.cos(math.pi * t))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def majority_vote_probs(logits: torch.Tensor, k: int = 7) -> np.ndarray:
    """
    logits: [T, C]（未 softmax 也可）
    返回：np.ndarray[int] [T] 平滑后的标签
    """
    assert k % 2 == 1, "k 必须是奇数"
    assert logits.dim() == 2, "logits 形状应为 [T, C]"
    T, C = logits.shape
    r = k // 2
    pad = F.pad(logits.T.unsqueeze(0), (r, r, 0, 0), mode="replicate").squeeze(0).T  # [T+2r, C]
    out = []
    for t in range(T):
        win_logits = pad[t:t+k]               # [k, C]
        probs = F.softmax(win_logits, dim=-1) # [k, C]
        avg_prob = probs.mean(dim=0)          # [C]
        out.append(int(torch.argmax(avg_prob)))
    return np.array(out, dtype=int)

def compute_class_balanced_weights(labels_np, num_classes, beta=0.999):
    """基于有效样本数(Cui et al.)计算类别权重，并归一化到均值=1"""
    counts = np.bincount(labels_np, minlength=num_classes).astype(np.int64)
    nz = counts[counts > 0]
    if len(nz) == 0:
        return torch.ones(num_classes, dtype=torch.float32), counts
    min_nz = nz.min()
    counts[counts == 0] = min_nz  # 防止除零/权重爆炸
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
    # 归一化使平均权重为 1
    weights = weights * (num_classes / np.sum(weights))
    return torch.tensor(weights, dtype=torch.float32), counts


class LabelSmoothingCrossEntropy(nn.Module):
    """兼容老版本 PyTorch 的 Label Smoothing + 可选 class weight"""
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
            true_dist.fill_(self.smoothing / (self.num_classes))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        loss_per_sample = (-true_dist * log_probs).sum(dim=1)  # [N]
        if self.weight is not None:
            w = self.weight.gather(0, target)
            loss_per_sample = loss_per_sample * w
        return loss_per_sample.mean()


def build_ce_loss(num_classes, class_weights_t, label_smoothing, device):
    """优先使用原生 CrossEntropyLoss(label_smoothing)，若不支持则回退自定义实现"""
    if label_smoothing > 0:
        try:
            return nn.CrossEntropyLoss(
                weight=(class_weights_t.to(device) if class_weights_t is not None else None),
                label_smoothing=label_smoothing
            ).to(device)
        except TypeError:
            # 老版本不支持 label_smoothing
            return LabelSmoothingCrossEntropy(
                num_classes=num_classes,
                smoothing=label_smoothing,
                weight=(class_weights_t.to(device) if class_weights_t is not None else None)
            ).to(device)
    else:
        return nn.CrossEntropyLoss(
            weight=(class_weights_t.to(device) if class_weights_t is not None else None)
        ).to(device)


def plot_and_save_curves(subject_id, train_losses, val_losses, train_accs, val_accs, gate_weights_history, save_dir):
    plt.figure(figsize=(18, 6))

    # 损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, 'b-', linewidth=2, label='训练损失')
    plt.plot(val_losses, 'r-', linewidth=2, label='验证损失')
    plt.title(f'训练和验证损失曲线 (被试 {subject_id})', fontsize=14, fontweight='bold')
    plt.xlabel('训练轮次', fontsize=12)
    plt.ylabel('损失值', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # 精度曲线
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, 'b-', linewidth=2, label='训练准确率')
    plt.plot(val_accs, 'r-', linewidth=2, label='验证准确率')
    plt.title(f'训练和验证准确率曲线 (被试 {subject_id})', fontsize=14, fontweight='bold')
    plt.xlabel('训练轮次', fontsize=12)
    plt.ylabel('准确率 (%)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # 门控权重曲线
    if len(gate_weights_history) > 0:
        epochs_hist = [x[0] for x in gate_weights_history]
        space_weights = [x[1] * 100 for x in gate_weights_history]
        time_dom_weights = [x[2] * 100 for x in gate_weights_history]
        time_freq_weights = [x[3] * 100 for x in gate_weights_history]

        plt.subplot(1, 3, 3)
        plt.plot(epochs_hist, space_weights, 'b-', linewidth=2, label='空间分支权重')
        plt.plot(epochs_hist, time_dom_weights, 'm-', linewidth=2, label='时域分支权重')
        plt.plot(epochs_hist, time_freq_weights, 'g-', linewidth=2, label='时频分支权重')
        plt.title(f'门控权重变化 (被试 {subject_id})', fontsize=14, fontweight='bold')
        plt.xlabel('训练轮次', fontsize=12)
        plt.ylabel('权重百分比 (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)

    plt.tight_layout()
    out_path = os.path.join(save_dir, f'training_curves_subject_{subject_id}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path


def plot_and_save_confusion(cm, subject_id, save_path):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, aspect='auto')
    plt.title(f'混淆矩阵 - 被试 {subject_id}', fontsize=16, fontweight='bold')
    cbar = plt.colorbar()
    cbar.set_label('样本数量', fontsize=12)
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


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    subject_accuracies = []
    all_subject_gate_weights = {}

    for subject_id in subject_range:
        print(f"\n处理受试者 {subject_id}...")
        try:
            emg_file = os.path.join(root_data, f'data/restimulus/reSegEnhance/{DATASET}_s{subject_id}SegEnhance.h5')
            feature_file = os.path.join(root_data, f'data/restimulus/Fea/{DATASET}_s{subject_id}feaEnhance.h5')
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

            # 划分集
            rep_vali = 2
            emg_train, emg_vali, emg_test, label_train, label_vali, label_test = get_threeSet(emg_data, labels, rep_arr, rep_vali)
            feature_train, feature_vali, feature_test, _, _, _ = get_threeSet(feature_data, labels, rep_arr, rep_vali)


            print("DB2/DB4 - EMG shape:", emg_train.shape)
            print("DB2/DB4 - Feature shape:", feature_train.shape)
            print("DB2/DB4 - Num classes:", np.max(label_train)+1)
        except Exception as e:
            print(f"数据加载或分割出错: {e}")
            continue  # 进入下一个受试者

        # 构建数据集与 DataLoader
        train_dataset = EMGDataset(emg_train, feature_train, label_train)
        val_dataset = EMGDataset(emg_vali, feature_vali, label_vali)
        test_dataset = EMGDataset(emg_test, feature_test, label_test)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
        test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

        # 通道数与时间长度
        if emg_train.ndim == 3 and emg_train.shape[-1] in (12, 16):  # (N, T, C)
            emg_channels = int(emg_train.shape[-1])
            time_steps   = int(emg_train.shape[1])
        else:  # 已是 (N, C, T)
            emg_channels = int(emg_train.shape[1])
            time_steps   = int(emg_train.shape[2])

        num_classes = int(np.max(label_train) + 1)

        # 构建模型
        model = ImprovedFeaAndEmg_V2(
            emg_channels=emg_channels,
            time_steps=time_steps,
            feature_dim=int(feature_train.shape[1]),
            num_classes=num_classes,
            dropout_rate=DROPOUT_RATE
        ).to(device)

        # >>> 在这里加一行（紧跟模型创建之后）
        model.set_ablation(ABLATED_BRANCH)

        # 去除通道注意力
        replaced_list = replace_channel_attention_with_identity(model)
        if len(replaced_list) == 0:
            print("未检测到可替换的通道注意力模块。若仍需移除，请把模型子模块类名贴给我。")
            unique_classes = sorted({m.__class__.__name__ for m in model.modules()})
            print("模型中出现的模块类名（前50个）：", unique_classes[:50])
        else:
            print("已替换以下通道注意力模块为 Identity：")
            for path, cls in replaced_list:
                print(f" - {path}: {cls}")

        # 打印参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型总参数量: {total_params:,} | 可训练参数量: {trainable_params:,}")

        # 损失函数
        class_weights_t = None
        if USE_CLASS_BALANCED_LOSS:
            class_weights_t, counts = compute_class_balanced_weights(label_train, num_classes, beta=CB_BETA)
            print("启用 Class-Balanced Loss（beta=%.3f）" % CB_BETA)
            top5_idx = np.argsort(class_weights_t.cpu().numpy())[::-1][:5]
            print("权重Top-5 类别与值：", [(int(i), float(class_weights_t[i])) for i in top5_idx])

        criterion = build_ce_loss(
            num_classes=num_classes,
            class_weights_t=class_weights_t,
            label_smoothing=LABEL_SMOOTHING,
            device=device
        )

        # 优化器与调度
        if USE_ADAMW:
            optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            print(f"优化器: AdamW(lr={LR}, weight_decay={WEIGHT_DECAY})")
        else:
            optimizer = Adam(model.parameters(), lr=LR)
            print(f"优化器: Adam(lr={LR})")
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8, min_lr=1e-5, verbose=True)

        # 早停
        early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE, min_delta=0.001)
        print(f"模型结构已更新（三分支，已去通道注意力），开始训练...")

        best_val_acc = 0.0
        best_model_state = None
        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        gate_weights_history = []  # (epoch_frac, space, time_dom, time_freq)

        for epoch in range(EPOCHS):
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            for batch_idx, (emg_inputs, fea_inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
                emg_inputs = emg_inputs.to(device)
                fea_inputs = fea_inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs, gate_weights = model(fea_inputs, emg_inputs, return_gate_weights=True)
                loss = criterion(outputs, labels)
                loss.backward()

                if GRAD_CLIP_NORM is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                if batch_idx % 10 == 0:
                    space_w = gate_weights[:, 0].mean().item()
                    time_dom_w = gate_weights[:, 1].mean().item()
                    time_freq_w = gate_weights[:, 2].mean().item()
                    gate_weights_history.append((
                        epoch + batch_idx / max(1, len(train_loader)),
                        space_w, time_dom_w, time_freq_w
                    ))

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
                    outputs, _ = model(fea_inputs, emg_inputs, return_gate_weights=True)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_acc = 100.0 * val_correct / max(1, val_total)
            val_losses.append(val_loss / max(1, len(val_loader)))
            val_accs.append(val_acc)

            scheduler.step(val_acc)
            print(f"Epoch {epoch+1}/{EPOCHS}, "
                  f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()

            # 早停
            if early_stopping(val_losses[-1], model):
                print(f"早停触发！在第 {epoch+1} 轮停止训练")
                break

        # 绘制并保存曲线
        curves_path = plot_and_save_curves(
            subject_id, train_losses, val_losses, train_accs, val_accs, gate_weights_history, RESULTS_DIR
        )

        # 保存最佳模型
        if best_model_state is None:
            best_model_state = model.state_dict().copy()
        best_model_path = os.path.join(RESULTS_DIR, f'best_model_subject_{subject_id}.pth')
        torch.save(best_model_state, best_model_path)
        print(f"最佳模型已保存到 {best_model_path}")

        # 测试
        model.load_state_dict(best_model_state)
        model.eval()
        all_preds, all_labels, all_gate_weights = [], [], []
        logits_list = []

        with torch.no_grad():
            for emg_inputs, fea_inputs, labels in test_loader:
                emg_inputs = emg_inputs.to(device)
                fea_inputs = fea_inputs.to(device)
                labels = labels.to(device)
                outputs, gate_weights = model(fea_inputs, emg_inputs, return_gate_weights=True)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_gate_weights.append(gate_weights.cpu().numpy())
                logits_list.append(outputs.detach().cpu())

        # 将 list 转为数组/张量
        all_logits = torch.cat(logits_list, dim=0)  # [N, C]
        all_labels_np = np.array(all_labels)
        all_preds_np = np.array(all_preds)

        # 多数投票（概率版）
        if USE_POST_PROCESSING:
            preds_for_metrics = majority_vote_probs(all_logits, k=7)  # k 可设 5/7/9
            print("已启用多数投票后处理（k=7）。")
        else:
            preds_for_metrics = all_preds_np

        # 用平滑后的预测计算准确率等
        test_acc = 100.0 * np.sum(preds_for_metrics == all_labels_np) / len(all_labels_np)
        subject_accuracies.append(test_acc)
        final_gate_weights = np.mean(np.concatenate(all_gate_weights, axis=0), axis=0)  # (3,)

        print(f"受试者 {subject_id} 测试准确率: {test_acc:.2f}%")
        print(
            f"测试集门控权重: 空间={final_gate_weights[0] * 100:.2f}%, 时域={final_gate_weights[1] * 100:.2f}%, 时频={final_gate_weights[2] * 100:.2f}%")
        all_subject_gate_weights[subject_id] = final_gate_weights

        # 混淆矩阵与报告也用平滑后的 preds
        cm = confusion_matrix(all_labels_np, preds_for_metrics)
        cm_path = os.path.join(RESULTS_DIR, f'confusion_matrix_subject_{subject_id}.png')
        plot_and_save_confusion(cm, subject_id, cm_path)

        report_str = classification_report(all_labels_np, preds_for_metrics, digits=2)
        report_path = os.path.join(RESULTS_DIR, f'classification_report_subject_{subject_id}.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_str)
        print(f"分类报告已保存到 {report_path}")
        print(f"训练曲线图已保存到 {curves_path}")

        # 汇总各被试
        if len(subject_accuracies) > 0:
            oa = float(np.mean(subject_accuracies))
            std = float(np.std(subject_accuracies))
            acc_max = float(np.max(subject_accuracies))
            acc_min = float(np.min(subject_accuracies))

            print("\n=== 总体评估结果 ===")
            print(f"处理的受试者数量: {len(subject_accuracies)}")
            print(f"总体分类准确度 (OA): {oa:.2f}%")
            print(f"准确度标准差: {std:.2f}%")
            print(f"最高准确度: {acc_max:.2f}%")
            print(f"最低准确度: {acc_min:.2f}%")

            # 平均门控权重
            if len(all_subject_gate_weights) > 0:
                gate_arr = np.stack(list(all_subject_gate_weights.values()), axis=0)  # [S, 3]
                gate_mean = gate_arr.mean(axis=0)  # (3,)
                space_pct = gate_mean[0] * 100.0
                time_pct = gate_mean[1] * 100.0
                tf_pct = gate_mean[2] * 100.0
                print(f"所有被试平均门控权重: 空间={space_pct:.2f}%, 时域={time_pct:.2f}%, 时频={tf_pct:.2f}%")
            else:
                print("所有被试平均门控权重: 无（未收集到门控权重）")

            # 写入 summary 文件
            summary_path = os.path.join(RESULTS_DIR, "summary.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write("=== 总体评估结果 ===\n")
                f.write(f"处理的受试者数量: {len(subject_accuracies)}\n")
                f.write(f"总体分类准确度 (OA): {oa:.2f}%\n")
                f.write(f"准确度标准差: {std:.2f}%\n")
                f.write(f"最高准确度: {acc_max:.2f}%\n")
                f.write(f"最低准确度: {acc_min:.2f}%\n")
                if len(all_subject_gate_weights) > 0:
                    f.write(f"所有被试平均门控权重: 空间={space_pct:.2f}%, 时域={time_pct:.2f}%, 时频={tf_pct:.2f}%\n")
                else:
                    f.write("所有被试平均门控权重: 无\n")
            print(f"汇总结果已保存到 {summary_path}")
        else:
            print("没有成功评估任何受试者，请检查数据路径或划分逻辑。")

if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print(">>> [ENTRY] Top-level exception:", repr(e))
        traceback.print_exc()

