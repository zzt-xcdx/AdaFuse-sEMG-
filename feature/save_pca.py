"""
save_pca.py
从已有的原始 EMG H5 文件重新拟合 PCA 并保存为 .pkl
不需要重跑 GetFeature.py 全流程
"""

import os
import gc
import h5py
import numpy as np
import joblib
from tqdm import tqdm
from scipy import signal
from stockwell import st
from sklearn.decomposition import IncrementalPCA

# ============================================================
# ✅ 配置（和 GetFeature.py 保持完全一致）
# ============================================================
ROOT_DATA   = 'D:/DB4'
TRAIN_REPS  = [1, 3, 4, 6]
TEST_REPS   = [2, 5]
COMPONENTS  = 64
BATCH_SIZE  = 256
ST_POOL     = 'time_mean'
ST_DS       = 2

# ============================================================
# ✅ 直接复制自 GetFeature.py 的函数（不要改）
# ============================================================

def get_rep_from_label(lbl):
    try:
        if isinstance(lbl, np.void) or (hasattr(lbl, 'dtype') and getattr(lbl.dtype, 'fields', None)):
            if 'repetition' in lbl.dtype.fields:
                return int(lbl['repetition'])
    except Exception:
        pass
    try:
        return int(lbl)
    except Exception:
        return None


def build_index_splits(labels):
    train_idx = []
    for i in range(len(labels)):
        rep = get_rep_from_label(labels[i])
        if rep in TRAIN_REPS:
            train_idx.append(i)
    return np.array(train_idx, dtype=np.int64)


def st_channel_features(channel, pool='time_mean', ds_factor=2):
    S = st.st(channel.astype(np.float32))
    M = np.abs(S)
    if pool == 'time_mean':
        v = M.mean(axis=1)
    elif pool == 'time_max':
        v = M.max(axis=1)
    elif pool == 'time_mean_log':
        v = np.log1p(M).mean(axis=1)
    else:
        v = M.reshape(-1)
    if ds_factor and ds_factor > 1 and v.ndim == 1 and v.shape[0] >= 2 * ds_factor:
        try:
            v = signal.decimate(v, ds_factor, ftype='fir', zero_phase=True)
        except Exception:
            v = v[::ds_factor]
    return v.astype(np.float32)


def compute_batch_features(emg_batch, pool='time_mean', ds_factor=2):
    B, T, C = emg_batch.shape
    feats = []
    for b in range(B):
        feat_per_sample = []
        for ch in range(C):
            v = st_channel_features(emg_batch[b, :, ch], pool=pool, ds_factor=ds_factor)
            feat_per_sample.append(v)
        feats.append(np.concatenate(feat_per_sample, axis=0))
    return np.vstack(feats)


def iterate_in_batches(indices, batch_size):
    for i in range(0, len(indices), batch_size):
        yield indices[i:i + batch_size]

# ============================================================
# ✅ 主流程
# ============================================================

def save_pca_for_subject(sid):
    in_file  = os.path.join(ROOT_DATA, 'data', 'restimulus', 'reSegEnhance',
                            f'DB4_s{sid}SegEnhance.h5')
    out_pkl  = os.path.join(ROOT_DATA, 'data', 'restimulus', 'Fea',
                            f'DB4_s{sid}feaEnhance_ipca.pkl')

    if not os.path.exists(in_file):
        print(f"❌ 找不到原始文件: {in_file}")
        return

    print(f"\n{'='*50}")
    print(f"被试 {sid}: 拟合 PCA")
    print(f"{'='*50}")

    with h5py.File(in_file, 'r') as f:
        emg_dset = f['emg']          # (N, T, C)
        labels   = f['label'][:]
        N        = emg_dset.shape[0]

        train_idx = build_index_splits(labels)
        print(f"训练样本数: {len(train_idx)}")

        ipca = IncrementalPCA(n_components=COMPONENTS,
                              batch_size=max(64, COMPONENTS))

        for idx_batch in tqdm(list(iterate_in_batches(train_idx, BATCH_SIZE)),
                              desc="partial_fit"):
            if np.all(np.diff(idx_batch) == 1):
                emg_batch = emg_dset[int(idx_batch[0]):int(idx_batch[-1])+1]
            else:
                emg_batch = np.stack([emg_dset[i] for i in idx_batch], axis=0)

            X = compute_batch_features(emg_batch, pool=ST_POOL, ds_factor=ST_DS)
            ipca.partial_fit(X)

            del emg_batch, X
            gc.collect()

    # ✅ 保存
    joblib.dump(ipca, out_pkl)
    print(f"✅ PCA 已保存: {out_pkl}")


if __name__ == "__main__":
    for sid in range(1, 11):
        save_pca_for_subject(sid)
    print("\n✅ 全部完成！")