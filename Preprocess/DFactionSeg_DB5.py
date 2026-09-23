import math
import h5py
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import nina_funcs as nf
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F

# 数据集配置
dataset = 'DB5'
root_data = 'D:/DB5'
subject_range = range(1, 11)  # DB5有10个被试
gestures = list(range(1, 53))  # 52个手势
endlabel = 52

train_reps = [1, 3, 4, 6]
test_reps = [2, 5]

# 滑动窗口参数
WIN_LEN = 40   # 200ms（200Hz采样率）
WIN_STRIDE = 8 # 40ms
CHANNEL = 16   # 16通道

# 分割函数适配16通道
def action_reseg(data, channel, endlabel):
    actionList = []
    begin = 0
    count = 0
    for aim in tqdm(range(1, len(data))):
        if (data.iloc[aim, channel] == endlabel + 1):
            break
        if (aim == len(data) - 1 or data.iloc[aim, channel] != data.iloc[aim - 1, channel]):
            end = aim
            rep_num = set(data.iloc[begin:end, channel + 1])
            if (data.iloc[begin, channel + 1] != math.floor((count % 12) / 2) + 1 or len(rep_num) != 1):
                count1 = math.floor((count % 12) / 2)+1
                data.loc[list(range(begin, end)), 'rerepetition'] = [count1 for _ in range(begin, end)]
            count = count + 1
            actionList.append(data[begin:end])
            begin = end
    return actionList

# 标准化与增强（可选，流程与原版一致）
def bnEnhancesegment(seglist, channel=16):
    BNdatalist = []
    for i in range(len(seglist)):
        temp_label = seglist[i].iloc[0, channel]
        temp_rep = seglist[i].iloc[0, channel + 1]
        if (temp_label == 0):
            continue
        if temp_rep in ([1, 3, 4, 6]):
            timedata = np.array(seglist[i].iloc[:, :channel].copy())
            data = []
            newdata = []
            for Numchannel in range(channel):
                datasample = timedata[:, Numchannel]
                # 这里可加数据增强，如tsaug等
                data.append(datasample)
                newdata.append(datasample)  # 不做增强，直接复制
            data = (np.array(data)).T
            newdata = (np.array(newdata)).T
            flag = True
            for iemg in ([data, newdata]):
                scaler = StandardScaler()
                Zscdata = scaler.fit_transform(iemg[:])
                BNdata = Zscdata
                if (flag):
                    bndata1 = BNdata
                    flag = False
                else:
                    bndata2 = BNdata
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

# 滑动窗口生成器适配16通道
def action_comb_generator(actionList, timeWindow, strideWindow, channel=16):
    for action in actionList:
        rep = int(action.values[0, channel + 1])
        stimulus = int(action.values[0, channel])
        if rep == 0 or stimulus == 0:
            continue
        length = math.floor((len(action) - timeWindow) / strideWindow) + 1
        for j in range(length):
            subImage = action.iloc[strideWindow * j:strideWindow * j + timeWindow, 0:channel]
            yield np.array(subImage), int(action.iloc[0, channel]), int(action.iloc[0, channel + 1])

# 线性插值上采样函数
def upsample_sequence(x, target_length=400):
    # x: (time=40, channels=16)
    x = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)  # (1, 40, 16)
    x_upsampled = F.interpolate(x.transpose(1, 2), size=target_length, mode='linear', align_corners=False)
    x_upsampled = x_upsampled.transpose(1, 2)
    return x_upsampled.squeeze(0).numpy()  # (400, 16)

for j in subject_range:
    input_file = os.path.join(root_data, f'data/restimulus/refilter/{dataset}_s{j}filter.h5')
    if not os.path.exists(input_file):
        print(f'跳过文件 {input_file} - 文件不存在')
        continue
    df = pd.read_hdf(input_file, 'df')
    actionList = action_reseg(df, CHANNEL, endlabel)
    bnList = bnEnhancesegment(actionList, CHANNEL)
    output_dir = os.path.join(root_data, 'data/restimulus/reSegEnhance')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{dataset}_s{j}SegEnhance.h5')
    with h5py.File(output_file, 'w') as f:
        # 先生成第一个窗口并插值，确定shape
        first_window, first_label, first_rep = next(action_comb_generator(bnList, WIN_LEN, WIN_STRIDE, channel=CHANNEL), (None, None, None))
        if first_window is None:
            print(f"未为文件 {input_file} 生成任何窗口。")
            continue
        first_window_upsampled = upsample_sequence(first_window, 400)
        emg_dset = f.create_dataset('emg', shape=(0,) + first_window_upsampled.shape, maxshape=(None,) + first_window_upsampled.shape, dtype='float32')
        label_dset = f.create_dataset('label', shape=(0,), maxshape=(None,), dtype='int')
        rep_dset = f.create_dataset('rep', shape=(0,), maxshape=(None,), dtype='int')
        emg_dset.resize((emg_dset.shape[0] + 1), axis=0)
        emg_dset[-1, :, :] = first_window_upsampled
        label_dset.resize((label_dset.shape[0] + 1), axis=0)
        label_dset[-1] = first_label
        rep_dset.resize((rep_dset.shape[0] + 1), axis=0)
        rep_dset[-1] = first_rep
        for window, label, rep in action_comb_generator(bnList, WIN_LEN, WIN_STRIDE, channel=CHANNEL):
            window_upsampled = upsample_sequence(window, 400)
            emg_dset.resize((emg_dset.shape[0] + 1), axis=0)
            emg_dset[-1, :, :] = window_upsampled
            label_dset.resize((label_dset.shape[0] + 1), axis=0)
            label_dset[-1] = label
            rep_dset.resize((rep_dset.shape[0] + 1), axis=0)
            rep_dset[-1] = rep
    print('******************' + dataset + '_s' + str(j) + '分割完成***********************')
    # 检查输出h5文件emg数据集shape
    with h5py.File(output_file, 'r') as f_check:
        print(f"{output_file} emg shape:", f_check['emg'].shape) 