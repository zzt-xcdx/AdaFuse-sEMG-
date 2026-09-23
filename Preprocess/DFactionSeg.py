
import os
import sys
import math
import h5py
import numpy as np
import pandas as pd
import tsaug
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# 若有自定义工具函数，保持原有搜索路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import nina_funcs as nf  # 若未使用也可保留，避免外部依赖出错


# =========================
# 数据集与窗口配置
# =========================
dataset = 'DB4'  # 可设为 'DB2' 或 'DB4'

if dataset == 'DB4':
    root_data = 'D:/DB4'
    subject_range = range(1, 11)     # DB4: s1-s10
    gestures = list(range(1, 53))    # 1-52
    endlabel = 52
else:
    root_data = 'D:/DB2'
    subject_range = range(1, 41)     # DB2: s1-s40
    gestures = list(range(1, 50))    # 1-49
    endlabel = 49

# 训练/测试重复（供增强和划分参考）
train_reps = [1, 3, 4, 6]
test_reps = [2, 5]

# 窗口参数（按需求调节）
FS = 2000           # 采样率 Hz
WINDOW_MS = 300     # 窗长：250 ms（精度优先）
STRIDE_MS = 60   # 步长：40 ms（重叠高、计算适中）

TIME_WINDOW = int(FS * WINDOW_MS / 1000)    # → 500 点
STRIDE_WINDOW = int(FS * STRIDE_MS / 1000)  # → 80 点


# =========================
# 功能函数
# =========================
def action_reseg(data: pd.DataFrame, channel: int, endlabel: int):
    """
    连续序列按标签变化切段，并重写 rerepetition，保证每段重复号一致且顺序正确。
    data: 带有列 [0:channel)=EMG, [channel]=stimulus, [channel+1]=rerepetition
    """
    actionList = []
    begin = 0
    count = 0  # 用于动作计数，纠正 rep 标签

    for aim in tqdm(range(1, len(data))):
        # 控制手势停止时间（大于 endlabel+1 视为记录结束）
        if data.iloc[aim, channel] == endlabel + 1:
            break

        # 标签发生变化或到达末尾，截取一个段
        if (aim == len(data) - 1) or (data.iloc[aim, channel] != data.iloc[aim - 1, channel]):
            end = aim
            rep_num = set(data.iloc[begin:end, channel + 1])

            # 如重复号混乱或不一致，则按顺序重写 rerepetition
            if (data.iloc[begin, channel + 1] != math.floor((count % 12) / 2) + 1) or (len(rep_num) != 1):
                count1 = math.floor((count % 12) / 2) + 1
                data.loc[list(range(begin, end)), 'rerepetition'] = [count1 for _ in range(begin, end)]

            count += 1
            actionList.append(data[begin:end])
            begin = end

    return actionList


def action_seg(data: pd.DataFrame, channel: int, endlabel: int):
    """
    与 action_reseg 类似，但写入列名为 repetition（保留以兼容旧代码）。
    """
    actionList = []
    begin = 0
    count = 0

    for aim in tqdm(range(1, len(data))):
        if data.iloc[aim, channel] == endlabel + 1:
            break

        if (aim == len(data) - 1) or (data.iloc[aim, channel] != data.iloc[aim - 1, channel]):
            end = aim
            rep_num = set(data.iloc[begin:end, channel + 1])

            if (data.iloc[begin, channel + 1] != math.floor((count % 12) / 2) + 1) or (len(rep_num) != 1):
                count1 = math.floor((count % 12) / 2) + 1
                data.loc[list(range(begin, end)), 'repetition'] = [count1 for _ in range(begin, end)]

            count += 1
            actionList.append(data[begin:end])
            begin = end

    return actionList


def uniform(actionList, channel: int):
    """
    对每个段内通道独立标准化（不改变外部列表引用）。
    """
    out = []
    for action in actionList:
        iemg = action.iloc[:, :channel].copy()
        scaler = preprocessing.StandardScaler()
        BNemg = scaler.fit_transform(iemg)
        action_copy = action.copy()
        action_copy.iloc[:, :channel] = BNemg
        out.append(action_copy)
    return out


def OneEnhanceEMG(data_1d: np.ndarray):
    """
    对单通道一维序列做 TimeWarp 增强。
    输入 data_1d 为一维数组（时间轴）。
    返回 (原始, 增强) 两条一维序列。
    """
    newdata = tsaug.TimeWarp(n_speed_change=1, max_speed_ratio=1.5, seed=123).augment(data_1d)
    return data_1d, newdata


def bnEnhancesegment(seglist, channel: int = 12):
    """
    段内标准化 + 训练重复的数据增强（时间扭曲）。
    - 训练重复 {1,3,4,6}：原始与增强各自独立标准化后再拼接
    - 测试重复 {2,5}：仅标准化，不增强
    输出：每段一个 DataFrame，列为 [0..channel-1]=EMG, 'stimulus', 'repetition'
    """
    BNdatalist = []
    for i in range(len(seglist)):
        temp_label = seglist[i].iloc[0, channel]
        temp_rep = seglist[i].iloc[0, channel + 1]

        # 跳过静息
        if temp_label == 0:
            continue

        if temp_rep in [1, 3, 4, 6]:
            timedata = np.array(seglist[i].iloc[:, :channel].copy())
            data = []
            newdata = []
            # 对每个通道分别增强，保持各通道间同步长度
            for Numchannel in range(channel):
                datasample, newdatasample = OneEnhanceEMG(timedata[:, Numchannel])
                data.append(datasample)
                newdata.append(newdatasample)
            data = np.array(data).T       # (time, channel)
            newdata = np.array(newdata).T

            # 原始和增强分别标准化，避免互相影响
            flag = True
            for iemg in [data, newdata]:
                scaler = StandardScaler()
                Zscdata = scaler.fit_transform(iemg[:])
                BNdata = Zscdata
                if flag:
                    bndata1 = BNdata
                    flag = False
                else:
                    bndata2 = BNdata

            # 拼接增强（倍增）
            BNdata = np.vstack((bndata1, bndata2))
            BNlabel = np.hstack([seglist[i].iloc[:, channel], seglist[i].iloc[:, channel]])
            BNrep = np.hstack([seglist[i].iloc[:, channel + 1], seglist[i].iloc[:, channel + 1]])

        else:
            iemg = np.array(seglist[i].iloc[:, :channel].copy())
            scaler = StandardScaler()
            Zscdata = scaler.fit_transform(iemg[:])
            BNdata = Zscdata
            BNlabel = np.array(seglist[i].iloc[:, channel])
            BNrep = np.array(seglist[i].iloc[:, channel + 1])

        myemg = pd.DataFrame(BNdata)
        myemg['stimulus'] = BNlabel
        myemg['repetition'] = BNrep
        BNdatalist.append(myemg)

    return BNdatalist


def action_comb_generator(actionList, timeWindow: int, strideWindow: int, channel: int = 12):
    """
    滑动窗口生成器。
    - 忽略 stimulus==0 或 rep==0 的段
    - 段长不足 timeWindow 的直接跳过
    逐窗返回：(window[np.float32, shape=(timeWindow, channel)], label[int], rep[int])
    """
    for action in actionList:
        rep = int(action.values[0, channel + 1])
        stimulus = int(action.values[0, channel])

        if rep == 0 or stimulus == 0:
            continue
        if len(action) < timeWindow:
            continue

        length = math.floor((len(action) - timeWindow) / strideWindow) + 1
        for j in range(length):
            sub = action.iloc[strideWindow * j:strideWindow * j + timeWindow, 0:channel]
            yield np.asarray(sub, dtype=np.float32), stimulus, rep


# =========================
# 主流程（示例：仅处理 s7、s8）
# =========================
if __name__ == "__main__":
    for j in subject_range:  # 如需全量处理，替换为: for j in subject_range:
        # 输入/输出路径
        input_file = os.path.join(root_data, f'data/restimulus/refilter/{dataset}_s{j}filter.h5')
        if not os.path.exists(input_file):
            print(f'跳过文件 {input_file} - 文件不存在')
            continue

        df = pd.read_hdf(input_file, 'df')

        # 1) 基于标签切段并纠正重复号
        actionList = action_reseg(df, 12, endlabel)

        # 2) 段内标准化 + 训练重复增强
        bnList = bnEnhancesegment(actionList, 12)

        # 3) 使用生成器滑窗并写入 HDF5（修复首窗重复+错标：只用一个生成器）
        output_dir = os.path.join(root_data, 'data/restimulus/reSegEnhance')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'{dataset}_s{j}SegEnhance.h5')

        with h5py.File(output_file, 'w') as f:
            gen = action_comb_generator(bnList, TIME_WINDOW, STRIDE_WINDOW)

            first = next(gen, (None, None, None))
            if first[0] is None:
                print(f"未为文件 {input_file} 生成任何窗口。")
                continue

            first_window, first_label, first_rep = first

            # 创建可变形状数据集
            emg_dset = f.create_dataset(
                'emg',
                shape=(0,) + first_window.shape,
                maxshape=(None,) + first_window.shape,
                dtype='float32'
            )
            label_dset = f.create_dataset('label', shape=(0,), maxshape=(None,), dtype='int')
            rep_dset = f.create_dataset('rep', shape=(0,), maxshape=(None,), dtype='int')

            # 先写首个样本
            emg_dset.resize(emg_dset.shape[0] + 1, axis=0)
            emg_dset[-1, :, :] = first_window
            label_dset.resize(label_dset.shape[0] + 1, axis=0)
            label_dset[-1] = int(first_label)
            rep_dset.resize(rep_dset.shape[0] + 1, axis=0)
            rep_dset[-1] = int(first_rep)

            # 再写剩余样本（继续使用同一个生成器）
            for window, label, rep in gen:
                emg_dset.resize(emg_dset.shape[0] + 1, axis=0)
                emg_dset[-1, :, :] = window
                label_dset.resize(label_dset.shape[0] + 1, axis=0)
                label_dset[-1] = int(label)
                rep_dset.resize(rep_dset.shape[0] + 1, axis=0)
                rep_dset[-1] = int(rep)

        print(f'****************** {dataset}_s{j} 分割完成 ******************')
