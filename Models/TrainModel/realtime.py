"""
realtime.py —— 逐窗口模拟实时推理
每次喂一个窗口进去，维护滑动投票队列，实时输出预测
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import re
import time
import warnings
warnings.filterwarnings("ignore", message="h5py is running against HDF5")

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from Models.PyTorchModels_improved import ImprovedFeaAndEmg_se_PyTorch
from Util.function import get_threeSet

# ============================================================
# 配置
# ============================================================
SUBJECT_ID     = 1
DATASET        = 'DB4'
root_data      = 'D:/DB4'
RESULTS_DIR    = f'results_report_finegate_{DATASET}_minimal_rollback'
rep_vali       = 2
ABLATED_BRANCH = 'None'
VOTE_K         = 7
SIM_DELAY      = 0.0   # 设0全速跑，设0.02模拟20ms间隔

# ============================================================
# 完整复制自 trainandtest.py
# ============================================================
def replace_channel_attention_with_identity(model):
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
            if any(p.search(cls_name) for p in patterns) or \
               name.lower() in ("eca", "eca_after", "se", "semodule", "se_layer", "channel_attention"):
                setattr(parent, name, nn.Identity())
                replaced.append((full_name, cls_name))
            else:
                _walk(child, full_name)
    _walk(model, "")

    def _replace_attr_path(obj, attr_path):
        parts = attr_path.split(".")
        parent = obj
        for p in parts[:-1]:
            if not hasattr(parent, p): return
            parent = getattr(parent, p)
        leaf = parts[-1]
        if hasattr(parent, leaf) and not isinstance(getattr(parent, leaf), nn.Identity):
            cls_name = getattr(parent, leaf).__class__.__name__
            setattr(parent, leaf, nn.Identity())
            replaced.append((attr_path, cls_name))

    _replace_attr_path(model, "space_branch.eca")
    _replace_attr_path(model, "time_domain_branch.eca_after")
    return replaced


def preprocess_emg_single(window: np.ndarray) -> torch.Tensor:
    """(T, C) → (1, C, T, 1)，与 EMGDataset 完全一致"""
    emg = window.T.copy()
    mean = emg.mean(axis=1, keepdims=True)
    std  = emg.std(axis=1,  keepdims=True) + 1e-6
    emg  = (emg - mean) / std
    emg  = emg[np.newaxis, :, :, np.newaxis]
    return torch.FloatTensor(emg)


# ============================================================
# 实时推理器（核心）
# ============================================================
class RealtimeInferencer:
    def __init__(self, model, device, vote_k=7):
        self.model      = model
        self.device     = device
        self.logits_buf = deque(maxlen=vote_k)

    def infer_one(self, emg_window: np.ndarray, fea_window: np.ndarray):
        """
        emg_window : (T, C)   —— 一个EMG窗口
        fea_window : (64,)    —— 对应的已处理特征
        返回: (pred_raw, pred_smooth, gate_weights)
        """
        # EMG 预处理
        emg_tensor = preprocess_emg_single(emg_window).to(self.device)      # (1,C,T,1)
        fea_tensor = torch.FloatTensor(fea_window[np.newaxis, :]).to(self.device)  # (1,64)

        with torch.no_grad():
            logits, gate_w = self.model(fea_tensor, emg_tensor, return_gate_weights=True)

        logits_cpu = logits.cpu().squeeze(0)          # (num_classes,)
        gate_cpu   = gate_w.cpu().squeeze(0).numpy()  # (3,)

        # 即时预测
        pred_raw = int(torch.argmax(logits_cpu))

        # 滑动窗口投票
        self.logits_buf.append(logits_cpu)
        buf = torch.stack(list(self.logits_buf), dim=0)  # (k, C)
        avg_prob = F.softmax(buf, dim=-1).mean(dim=0)
        pred_smooth = int(torch.argmax(avg_prob))

        return pred_raw, pred_smooth, gate_cpu


# ============================================================
# 主流程
# ============================================================
def run_realtime(subject_id: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # -------- 读取数据 --------
    emg_file = os.path.join(root_data, f'data/restimulus/reSegEnhance/{DATASET}_s{subject_id}SegEnhance.h5')
    fea_file = os.path.join(root_data, f'data/restimulus/Fea/{DATASET}_s{subject_id}feaEnhance.h5')

    with h5py.File(emg_file, 'r') as f:
        emg_data = f['emg'][:]
        labels   = f['label'][:]
        rep_arr  = f['rep'][:]

    with h5py.File(fea_file, 'r') as f:
        fea_data = f['features'][:]

    # -------- 划分测试集（与 trainandtest.py 完全一致）--------
    _, _, emg_test, _, _, label_test = get_threeSet(emg_data, labels, rep_arr, rep_vali)
    _, _, fea_test, _, _, _          = get_threeSet(fea_data,  labels, rep_arr, rep_vali)

    N, T, C  = emg_test.shape
    feat_dim = fea_test.shape[1]
    num_classes = int(label_test.max()) + 1
    print(f"测试窗口数: {N} | T={T} | C={C} | feat_dim={feat_dim} | 类别数={num_classes}")

    # -------- 构建模型 --------
    model = ImprovedFeaAndEmg_se_PyTorch(
        emg_channels=C,
        time_steps=T,
        feature_dim=feat_dim,
        num_classes=num_classes,
        dropout_rate=0.0
    ).to(device)
    model.set_ablation(ABLATED_BRANCH)
    replace_channel_attention_with_identity(model)

    model_path = os.path.join(RESULTS_DIR, f'best_model_subject_{subject_id}.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ 模型加载成功: {model_path}")

    # -------- 逐窗口实时推理 --------
    inferencer = RealtimeInferencer(model, device, vote_k=VOTE_K)

    print(f"\n🚀 逐窗口实时推理开始（SIM_DELAY={SIM_DELAY*1000:.0f}ms）\n")
    print(f"{'帧':>6} | {'真实':>4} | {'即时':>4} | {'平滑':>4} | {'空间':>6} {'时域':>6} {'时频':>6} | {'延迟':>7}")
    print("-" * 62)

    preds_raw    = []
    preds_smooth = []
    latencies    = []

    for i in range(N):
        emg_win = emg_test[i]   # (T, C)  ← 模拟传感器推送一帧
        fea_win = fea_test[i]   # (64,)   ← 对应特征

        t0 = time.perf_counter()
        pred_raw, pred_smooth, gate_w = inferencer.infer_one(emg_win, fea_win)
        lat_ms = (time.perf_counter() - t0) * 1000

        preds_raw.append(pred_raw)
        preds_smooth.append(pred_smooth)
        latencies.append(lat_ms)

        if i % 50 == 0 or i < 10:
            gt = int(label_test[i])
            correct_mark = "✅" if pred_smooth == gt else "❌"
            print(f"{i:6d} | {gt:4d} | {pred_raw:4d} | {pred_smooth:4d} {correct_mark} | "
                  f"{gate_w[0]*100:5.1f}% {gate_w[1]*100:5.1f}% {gate_w[2]*100:5.1f}% | "
                  f"{lat_ms:6.2f}ms")

        if SIM_DELAY > 0:
            time.sleep(SIM_DELAY)

    # -------- 汇总 --------
    preds_raw    = np.array(preds_raw)
    preds_smooth = np.array(preds_smooth)
    acc_raw    = 100.0 * np.sum(preds_raw    == label_test) / N
    acc_smooth = 100.0 * np.sum(preds_smooth == label_test) / N
    avg_lat    = np.mean(latencies)
    p95_lat    = np.percentile(latencies, 95)

    print(f"\n{'='*55}")
    print(f"✅ 受试者 {subject_id} 实时推理完成！")
    print(f"   即时准确率  (无平滑) : {acc_raw:.2f}%")
    print(f"   平滑准确率 (k={VOTE_K}投票): {acc_smooth:.2f}%")
    print(f"   平均推理延迟: {avg_lat:.2f}ms  |  P95延迟: {p95_lat:.2f}ms")
    print(f"{'='*55}")


if __name__ == "__main__":
    run_realtime(SUBJECT_ID)