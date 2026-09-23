"""
realtime_random.py —— 随机输入手势模拟实时推理
随机从测试集抽窗口，模拟用户随机做手势，逐帧输出预测结果
"""
import sys, os, re, time, warnings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
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
SIM_DELAY      = 0.6    # 每帧间隔0.6s，方便肉眼观察
NUM_GESTURES   = 20     # 随机模拟多少个手势

# 手势名称（DB4的53类，没有的话用编号代替）
GESTURE_NAMES  = {}     # 留空则显示编号，例如 {0: "握拳", 1: "张开", ...}

# ============================================================
# 工具函数
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
               name.lower() in ("eca","eca_after","se","semodule","se_layer","channel_attention"):
                setattr(parent, name, nn.Identity())
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
            setattr(parent, leaf, nn.Identity())
    _replace_attr_path(model, "space_branch.eca")
    _replace_attr_path(model, "time_domain_branch.eca_after")


def preprocess_emg_single(window):
    emg = window.T.copy()
    mean = emg.mean(axis=1, keepdims=True)
    std  = emg.std(axis=1,  keepdims=True) + 1e-6
    emg  = (emg - mean) / std
    return torch.FloatTensor(emg[np.newaxis, :, :, np.newaxis])


def gesture_name(label):
    return GESTURE_NAMES.get(label, f"手势#{label:02d}")


# ============================================================
# 实时推理器
# ============================================================
class RealtimeInferencer:
    def __init__(self, model, device, vote_k=7):
        self.model      = model
        self.device     = device
        self.logits_buf = deque(maxlen=vote_k)

    def reset(self):
        """换手势时清空投票缓冲区"""
        self.logits_buf.clear()

    def infer_one(self, emg_window, fea_window):
        emg_t = preprocess_emg_single(emg_window).to(self.device)
        fea_t = torch.FloatTensor(fea_window[np.newaxis, :]).to(self.device)

        with torch.no_grad():
            logits, gate_w = self.model(fea_t, emg_t, return_gate_weights=True)

        logits_cpu = logits.cpu().squeeze(0)
        gate_cpu   = gate_w.cpu().squeeze(0).numpy()

        pred_raw = int(torch.argmax(logits_cpu))

        self.logits_buf.append(logits_cpu)
        avg_prob    = F.softmax(torch.stack(list(self.logits_buf)), dim=-1).mean(0)
        pred_smooth = int(torch.argmax(avg_prob))
        confidence  = float(avg_prob.max()) * 100

        return pred_raw, pred_smooth, confidence, gate_cpu


# ============================================================
# 主流程
# ============================================================
def run_random(subject_id: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- 读数据 --------
    emg_file = os.path.join(root_data, f'data/restimulus/reSegEnhance/{DATASET}_s{subject_id}SegEnhance.h5')
    fea_file = os.path.join(root_data, f'data/restimulus/Fea/{DATASET}_s{subject_id}feaEnhance.h5')

    with h5py.File(emg_file, 'r') as f:
        emg_data = f['emg'][:]
        labels   = f['label'][:]
        rep_arr  = f['rep'][:]
    with h5py.File(fea_file, 'r') as f:
        fea_data = f['features'][:]

    _, _, emg_test, _, _, label_test = get_threeSet(emg_data, labels, rep_arr, rep_vali)
    _, _, fea_test, _, _, _          = get_threeSet(fea_data,  labels, rep_arr, rep_vali)

    N, T, C  = emg_test.shape
    feat_dim = fea_test.shape[1]
    num_classes = int(label_test.max()) + 1

    # -------- 按类别建索引，方便随机抽 --------
    class_indices = {}
    for idx, lb in enumerate(label_test):
        class_indices.setdefault(int(lb), []).append(idx)
    available_classes = sorted(class_indices.keys())
    print(f"✅ 测试集就绪：{N}个窗口，{num_classes}类，{len(available_classes)}类有测试样本\n")

    # -------- 加载模型 --------
    model = ImprovedFeaAndEmg_se_PyTorch(
        emg_channels=C, time_steps=T,
        feature_dim=feat_dim, num_classes=num_classes,
        dropout_rate=0.0
    ).to(device)
    model.set_ablation(ABLATED_BRANCH)
    replace_channel_attention_with_identity(model)
    model.load_state_dict(torch.load(
        os.path.join(RESULTS_DIR, f'best_model_subject_{subject_id}.pth'),
        map_location=device
    ))
    model.eval()
    print("✅ 模型加载成功\n")

    inferencer = RealtimeInferencer(model, device, vote_k=VOTE_K)

    # -------- 随机手势模拟 --------
    print("=" * 60)
    print(f"  🎲 随机手势模拟开始！共 {NUM_GESTURES} 个手势")
    print("=" * 60)

    correct = 0
    for g in range(NUM_GESTURES):

        # 随机选一个类别，再随机抽一个窗口
        gt_class = int(np.random.choice(available_classes))
        idx      = int(np.random.choice(class_indices[gt_class]))

        inferencer.reset()   # 每个新手势清空投票缓冲

        emg_win = emg_test[idx]
        fea_win = fea_test[idx]

        print(f"\n[{g+1:02d}/{NUM_GESTURES}] 🖐  输入: {gesture_name(gt_class)}")

        t0 = time.perf_counter()
        pred_raw, pred_smooth, conf, gate_w = inferencer.infer_one(emg_win, fea_win)
        lat_ms = (time.perf_counter() - t0) * 1000

        hit = pred_smooth == gt_class
        correct += hit
        mark = "✅ 正确" if hit else "❌ 错误"

        print(f"         预测: {gesture_name(pred_smooth)}  ({conf:.1f}% 置信度)  {mark}")
        print(f"         门控: 空间{gate_w[0]*100:.1f}% | 时域{gate_w[1]*100:.1f}% | 时频{gate_w[2]*100:.1f}%  |  推理{lat_ms:.1f}ms")

        if SIM_DELAY > 0:
            time.sleep(SIM_DELAY)

    # -------- 汇总 --------
    print(f"\n{'='*60}")
    print(f"  🎯 随机手势模拟结束！")
    print(f"  正确率: {correct}/{NUM_GESTURES} = {correct/NUM_GESTURES*100:.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_random(SUBJECT_ID)