import os
import sys
import warnings
import argparse
import gc
import h5py
import numpy as np
from tqdm import tqdm
from scipy import signal
from stockwell import st
from sklearn.decomposition import IncrementalPCA

# -----------------------------
# 配置区
# -----------------------------
# 训练/测试重复编号（按你的项目约定）
TRAIN_REPS = [1, 3, 4, 6]
TEST_REPS = [2, 5]

# 默认数据根目录
DEFAULT_ROOT = 'D:/DB4'


# -----------------------------
# 工具函数
# -----------------------------
def get_rep_from_label(lbl):
    """
    从 label 条目提取 repetition 值，兼容结构化dtype或简单整数
    """
    # 结构化 dtype
    try:
        if isinstance(lbl, np.void) or (hasattr(lbl, 'dtype') and getattr(lbl.dtype, 'fields', None)):
            if 'repetition' in lbl.dtype.fields:
                return int(lbl['repetition'])
    except Exception:
        pass
    # 整数或可转 int
    try:
        return int(lbl)
    except Exception:
        return None


def build_index_splits(labels):
    train_idx, test_idx = [], []
    for i in range(len(labels)):
        rep = get_rep_from_label(labels[i])
        if rep in TRAIN_REPS:
            train_idx.append(i)
        elif rep in TEST_REPS:
            test_idx.append(i)
    return np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)


def st_channel_features(channel, pool='time_mean', ds_factor=2):
    """
    对单通道信号做 Stockwell 变换，输出降维后的特征向量
    - pool='time_mean': 沿时间维求均值，只保留频率维
    - ds_factor: 频率维降采样（>=1，默认2）
    """
    S = st.st(channel.astype(np.float32))  # 形状约为 (freq, time)
    M = np.abs(S)

    if pool == 'time_mean':
        v = M.mean(axis=1)  # 仅保留频率轴
    elif pool == 'time_max':
        v = M.max(axis=1)
    elif pool == 'time_mean_log':
        v = np.log1p(M).mean(axis=1)
    else:
        # 不池化：展开全部（强烈不建议，会非常大）
        v = M.reshape(-1)

    # 频率轴降采样
    if ds_factor and ds_factor > 1 and v.ndim == 1 and v.shape[0] >= 2 * ds_factor:
        try:
            v = signal.decimate(v, ds_factor, ftype='fir', zero_phase=True)
        except Exception:
            # 回退到简单切片降采样
            v = v[::ds_factor]

    return v.astype(np.float32)


def compute_batch_features(emg_batch, pool='time_mean', ds_factor=2):
    """
    对一个批次的 (B, T, C) 数据计算特征
    返回形状：(B, F)，其中 F 取决于池化/降采样后每通道长度 x 通道数
    """
    B, T, C = emg_batch.shape
    feats = []
    for b in range(B):
        feat_per_sample = []
        # emg_batch[b]: (T, C)
        for ch in range(C):
            v = st_channel_features(emg_batch[b, :, ch], pool=pool, ds_factor=ds_factor)
            feat_per_sample.append(v)
        feat_per_sample = np.concatenate(feat_per_sample, axis=0)
        feats.append(feat_per_sample)
    return np.vstack(feats)


def iterate_indices_in_batches(indices, batch_size):
    n = len(indices)
    for i in range(0, n, batch_size):
        yield indices[i:i + batch_size]


def ensure_output_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


# -----------------------------
# 主流程（流式双遍：fit -> transform）
# -----------------------------
def process_subject(input_file, output_file, components=64, batch_size=256, pool='time_mean', ds_factor=2, limit=None):
    """
    对一个被试进行处理：
    1) 从 input_file 读取 emg/label
    2) 构建训练/测试索引
    3) 第1遍：仅训练索引，分批计算特征 -> partial_fit(IncrementalPCA)
    4) 第2遍：全部样本分批计算特征 -> ipca.transform -> H5写出特征 + 标签
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    with h5py.File(input_file, 'r') as f_in:
        if 'emg' not in f_in or 'label' not in f_in:
            raise KeyError("输入文件缺少必要的数据集('emg'或'label')")
        emg_dset = f_in['emg']     # HDF5 dataset (N, T, C)
        labels = f_in['label'][:]  # 一次性读出标签（通常较小）

        N, T, C = emg_dset.shape
        if limit is not None and limit > 0:
            N = min(N, int(limit))
        print(f"数据形状: ({N}, {T}, {C})")
        print(f"标签数量: {len(labels)}")

        # 索引
        all_indices = np.arange(N, dtype=np.int64)
        train_idx, test_idx = build_index_splits(labels[:N])

        if len(train_idx) == 0:
            raise ValueError("训练索引集合为空，无法拟合 PCA")

        print(f"训练样本数: {len(train_idx)}，测试样本数: {len(test_idx)}")

        # 第1遍：拟合增量PCA（仅训练集）
        print("\n1) 拟合增量 PCA（训练集分批）...")
        ipca = IncrementalPCA(n_components=components, batch_size=max(64, components))

        for idx_batch in tqdm(list(iterate_indices_in_batches(train_idx, batch_size)), desc="PCA partial_fit"):
            # HDF5 支持按索引数组读取，但效率不如连续片段。
            # 为兼顾简单与性能，这里分两种情况：
            if np.all(np.diff(idx_batch) == 1):
                # 连续片段
                start, end = int(idx_batch[0]), int(idx_batch[-1]) + 1
                emg_batch = emg_dset[start:end]
            else:
                # 非连续：逐条拼接（代价略高，但训练集通常分散）
                emg_batch = np.stack([emg_dset[i] for i in idx_batch], axis=0)

            X = compute_batch_features(emg_batch, pool=pool, ds_factor=ds_factor)
            if np.any(~np.isfinite(X)):
                raise ValueError("训练特征包含 NaN/Inf")

            ipca.partial_fit(X)

            del emg_batch, X
            gc.collect()

        # 第1遍结束后，立刻保存 PCA
        import joblib
        joblib.dump(ipca, output_file.replace('.h5', '_ipca.pkl'))
        print(f"PCA 已保存到: {output_file.replace('.h5', '_ipca.pkl')}")

        # 第2遍：全部样本转换并写出
        print("\n2) 转换全部样本并写出 H5 ...")
        ensure_output_dir(output_file)
        with h5py.File(output_file, 'w') as f_out:
            feat_dset = f_out.create_dataset(
                'features',
                shape=(N, components),
                dtype='float32',
                chunks=(min(1024, N), components),
                compression="gzip"
            )
            f_out.create_dataset('label', data=labels[:N])

            write_pos = 0
            for i in tqdm(range(0, N, batch_size), desc="Transform+Write"):
                j = min(i + batch_size, N)
                emg_batch = emg_dset[i:j]  # 连续片段读取
                X = compute_batch_features(emg_batch, pool=pool, ds_factor=ds_factor)
                Z = ipca.transform(X).astype(np.float32)

                feat_dset[write_pos:write_pos + (j - i)] = Z
                write_pos += (j - i)

                del emg_batch, X, Z
                gc.collect()

        print(f"结果已保存到: {output_file}")


# -----------------------------
# 命令行接口
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="EMG 特征抽取：S变换 + 流式增量PCA（无超大临时文件）")
    p.add_argument("--force", action="store_true", help="忽略已完成输出，强制重跑")
    p.add_argument("--subjects", type=str, default="", help="仅处理指定被试，逗号分隔，如: 1,3,7")
    p.add_argument("--root-data", type=str, default=None, help="覆盖数据根目录，默认使用脚本中的 DEFAULT_ROOT")
    p.add_argument("--ignore-h5py-warning", action="store_true", help="忽略 h5py/HDF5 版本警告")
    p.add_argument("--components", type=int, default=64, help="PCA 输出维度")
    p.add_argument("--batch-size", type=int, default=256, help="处理批大小（越大越快，但占内存多）")
    p.add_argument("--st-pool", type=str, default="time_mean", choices=["time_mean", "time_max", "time_mean_log", "none"],
                   help="S 变换的时间维池化方式（none 会非常大，不建议）")
    p.add_argument("--st-ds", type=int, default=2, help="频率轴降采样倍数（>=1；1 表示不降采样）")
    p.add_argument("--limit", type=int, default=None, help="仅处理前 N 个样本用于快速验证")
    return p.parse_args()


def main():
    args = parse_args()

    if args.ignore_h5py_warning:
        warnings.filterwarnings("ignore",
                                message="h5py is running against HDF5",
                                category=UserWarning,
                                module="h5py")

    root_data = args.root_data or DEFAULT_ROOT
    np.random_seed = 42  # 可按需固定随机性

    # 输出目录
    out_dir = os.path.join(root_data, 'data', 'restimulus', 'Fea')
    os.makedirs(out_dir, exist_ok=True)

    # 计算完成的被试
    completed = []
    for sid in range(1, 11):
        out_file = os.path.join(out_dir, f'DB4_s{sid}feaEnhance.h5')
        if os.path.exists(out_file):
            try:
                with h5py.File(out_file, 'r') as f:
                    if 'features' in f and 'label' in f and f['features'].shape[0] > 0:
                        completed.append(sid)
            except Exception:
                pass

    if completed:
        print(f"已完成的被试: {completed}")
    else:
        print("没有找到已完成的被试")

    # 解析被试集合
    if args.subjects.strip():
        targets = sorted({int(s.strip()) for s in args.subjects.split(',') if s.strip()})
        print(f"指定仅处理的被试: {targets}")
    else:
        targets = list(range(1, 11))

    if not args.force:
        targets = [s for s in targets if s not in completed]

    print("\n从被试 1 开始处理...")
    print(f"本次待处理的被试: {targets}")

    for sid in targets:
        print("\n" + "=" * 50)
        print(f"处理被试 {sid}")
        print("=" * 50)

        in_file = os.path.join(root_data, 'data', 'restimulus', 'reSegEnhance', f'DB4_s{sid}SegEnhance.h5')
        out_file = os.path.join(out_dir, f'DB4_s{sid}feaEnhance.h5')

        try:
            process_subject(
                input_file=in_file,
                output_file=out_file,
                components=args.components,
                batch_size=args.batch_size,
                pool=args.st_pool,
                ds_factor=args.st_ds,
                limit=args.limit
            )
        except Exception as e:
            print(f"\n处理被试 {sid} 时出错: {type(e).__name__}: {e}")
            # 出错时不留半成品
            if os.path.exists(out_file):
                try:
                    os.remove(out_file)
                    print(f"已清理不完整输出: {out_file}")
                except Exception:
                    pass

    print("\n处理完成!")


if __name__ == "__main__":
    main()