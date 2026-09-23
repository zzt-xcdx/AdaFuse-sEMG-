import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy import signal
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix
import os
import pywt
import datetime
from tqdm import tqdm
from sklearn.decomposition import PCA
import scipy as sp


# DWPT分解
def emg_dwpt(signal, wavelet_name='db1'):
    wavelet_level = int(np.log2(len(signal)))
    wp = pywt.WaveletPacket(signal, wavelet_name, mode='sym')
    coeffs = []
    level_coeff = wp.get_level(wavelet_level)
    for i in range(len(level_coeff)):
        coeffs.append(level_coeff[i].data)
    coeffs = np.array(coeffs)
    coeffs = coeffs.flatten()
    return coeffs


def fft(data):
    return np.fft.fft(data)


def psd(data):
    return np.abs(np.fft.fft(data)) ** 2


''' ---时间域的特征分隔线--- '''


def mav(data):
    return np.mean(np.abs(data))


def iemg(data):
    return np.sum(np.abs(data))


def wl(data):
    return np.sum(abs(np.diff(data)))


def rms(data):
    return np.sqrt(np.mean(data ** 2))


def hist(data, nbins=20):
    histsig, bin_edges = np.histogram(data, bins=nbins)
    return tuple(histsig)


def entropy(data):
    pk = sp.stats.rv_histogram(np.histogram(data, bins=20)).pdf(data)
    return sp.stats.entropy(pk)


def kurtosis(data):
    return sp.stats.kurtosis(data)


##np.diff后一个元素减去前一个 np.sign函数返回的是一个由 1 和 -1 组成的数组
def zero_cross(data):
    return len(np.where(np.diff(np.sign(data)))[0])


def min(data):
    return np.min(data)


def max(data):
    return np.max(data)


def var(data):
    return np.var(data)


def mean(data):
    return np.mean(data)


def median(data):
    return np.median(data)


def wamp(signal, th=5e-3):
    x = abs(np.diff(signal))
    umbral = x >= th
    return np.sum(umbral)


def ssc(signal, threshold=1e-5):
    signal = np.array(signal)
    temp = [(signal[i] - signal[i - 1]) * (signal[i] - signal[i + 1]) for i in range(1, signal.shape[0] - 1, 1)]
    temp = np.array(temp)

    temp = temp[temp >= threshold]
    return temp.shape[0]


def aac(signal):
    signal = np.array(signal)
    length = signal.shape[0]
    wl = [abs(signal[i + 1] - signal[i]) for i in range(length - 1)]
    return np.mean(wl)


def feature_extractor(features, shape, data):
    """
    从形状为 (样本数, 时间步长, 通道数) 的数据中提取时域特征。
    在每个通道的数据 (形状为 (时间步长,)) 上计算 features 列表中的时域特征。
    """
    n_samples = shape[0]  # 样本数
    n_time_steps = data.shape[1]  # 时间步长 (例如 400)
    n_channels = data.shape[2]  # 通道数 (例如 12)
    n_features_per_type = len(features)  # 每种时域特征的数量 (例如 6)

    # 最终特征数量 = 通道数 * 每种时域特征的数量 = 12 * 6 = 72
    n_total_features = n_channels * n_features_per_type
    features_arr = np.zeros((n_samples, n_total_features))

    # 外层循环遍历每个样本
    for sample in range(n_samples):
        # 内层循环遍历每个通道
        for channel in range(n_channels):
            # 获取某个样本在某个通道上的所有时间步数据
            # data[sample, :, channel] 提取的是：
            # 对于第 sample 个样本 (sample 索引)
            # 获取所有时间步 (:) 的数据
            # 在第 channel 个通道 (channel 索引) 上的值
            # signal 的形状将是 (时间步长,)，即 (400,)
            signal = data[sample, :, channel]

            # 计算当前通道上所有选定时域特征的值
            temp = []
            for feature_function in features:
                # 在形状为 (400,) 的信号片段上调用特征计算函数
                temp.append(feature_function(signal))

            # 将计算出的每个通道的特征存储到 features_arr 的正确位置
            # features_arr[sample, ...] 表示存储第 sample 个样本的特征
            # channel * n_features_per_type 计算当前通道的特征在 features_arr 中的起始列索引
            # (channel + 1) * n_features_per_type 计算当前通道的特征在 features_arr 中的结束列索引
            # 这样，每个通道的特征会依次拼接起来
            features_arr[sample, channel * n_features_per_type:(channel + 1) * n_features_per_type] = temp

    return features_arr


# 找到这个函数定义，将整个函数体替换为以下代码
def frequency_features_extractor(shape, data, fs=2000):
    """
    从形状为 (样本数, 时间步长, 通道数) 的数据中提取频率域特征。
    在每个通道的数据 (形状为 (时间步长,)) 上计算选定的频率域特征。
    """
    # 确保导入了频率域特征计算所需的函数
    # 这里的导入语句假设这些函数定义在 Util/feature_set.py 中
    # 如果您的项目结构不同，请调整此处的导入路径或函数名
    from Util.feature_set import spectrum, frequency_ratio, mean_freq, median_freq

    n_samples = shape[0]  # 样本数
    n_time_steps = data.shape[1]  # 时间步长 (例如 400)
    n_channels = data.shape[2]  # 通道数 (例如 12)
    n_features_per_type = 5  # 每种频率特征的数量 (5)

    # 最终特征数量 = 通道数 * 每种频率特征的数量 = 12 * 5 = 60
    n_total_features = n_channels * n_features_per_type
    # 创建一个数组用于存储所有频率特征，形状为 (样本数, 总特征数量)
    features_arr = np.zeros((n_samples, n_total_features))

    # 使用列表临时存储计算出的每个样本、每个通道的特征，方便后续重塑和拼接
    fr_list = []
    mnp_list = []
    mnf_list = []
    mdf_list = []
    pkf_list = []

    # tqdm 用于显示循环进度条，如果不需要可以移除
    for i in tqdm(list(range(n_samples))):
        for j in range(n_channels): # 在通道维度上循环 (12 次)
            # 获取当前样本在当前通道的所有时间步数据
            # signal 的形状将是 (时间步长,)，即 (400,)
            signal = data[i, :, j]

            # 计算FFT和PSD
            # 确保 fft 和 psd 函数在 nina_funcs.py 文件中有定义并能正确处理形状为 (400,) 的数据
            fft_data = fft(signal)
            psd_data = psd(signal)

            # 计算频率特征
            # 1. 频率范围 (fr)
            fr = np.max(np.abs(fft_data)) - np.min(np.abs(fft_data))

            # 2. 平均功率 (mnp)
            mnp = np.mean(psd_data)

            # 3. 平均频率 (mnf)
            # freqs 是与 psd_data 对应的频率值
            freqs = np.fft.fftfreq(len(signal), 1/fs) # 修正：np.fft.fftfreq 需要采样间隔，这里假设采样率为 fs

            # 防止 psd_data 求和为零导致分母为零
            if np.sum(psd_data) == 0:
                 mnf = 0
            else:
                 mnf = np.sum(freqs * psd_data) / np.sum(psd_data)

            # 4. 中值频率 (mdf)
            # 计算功率谱的累积和
            cumsum = np.cumsum(psd_data)
            # 防止 cumsum 为空或最后一个元素（总功率）为零
            if cumsum.size == 0 or cumsum[-1] == 0:
                 mdf = 0
            else:
                 # 找到累积功率大于等于总功率一半的第一个索引
                 median_freq_index = np.where(cumsum >= cumsum[-1]/2)[0][0]
                 # 对应的频率就是中值频率
                 mdf = freqs[median_freq_index]


            # 5. 峰值频率 (pkf)
            # 防止 psd_data 为空
            if psd_data.size == 0:
                pkf = 0
            else:
                # 找到功率谱中最大值的索引，对应的频率就是峰值频率
                pkf = freqs[np.argmax(psd_data)]

            # 将计算出的每个样本、每个通道的 5 个频率特征存储到临时列表中
            fr_list.append(fr)
            mnp_list.append(mnp)
            mnf_list.append(mnf)
            mdf_list.append(mdf)
            pkf_list.append(pkf)

        # 将临时列表重塑为 (样本数, 通道数) 的形状
        # 例如，fr_list 包含 n_samples * n_channels 个值
        # 重塑为 (n_samples, n_channels) 后，每一列代表一个通道的 fr 特征
        fr_arr = np.array(fr_list).reshape((n_samples, n_channels))
        mnp_arr = np.array(mnp_list).reshape((n_samples, n_channels))
        mnf_arr = np.array(mnf_list).reshape((n_samples, n_channels))
        mdf_arr = np.array(mdf_list).reshape((n_samples, n_channels))
        pkf_arr = np.array(pkf_list).reshape((n_samples, n_channels))

        # 沿着列 (axis=1) 拼接所有频率特征数组
        # 拼接后形状为 (样本数, 通道数 * 5)，即 (样本数, 60)
        features_arr = np.concatenate([fr_arr, mnp_arr, mnf_arr, mdf_arr, pkf_arr], axis=1)

        return features_arr



'''数据处理分割线'''


def get_data(path, file):
    mat = loadmat(os.path.join(path, file))
    data = pd.DataFrame(mat['emg'])
    df_len = np.min([len(mat['emg']), len(mat['stimulus'])])
    data = pd.DataFrame(mat['emg'][:df_len])
    data['stimulus'] = mat['stimulus'][:df_len]
    data['repetition'] = mat['repetition'][:df_len]
    return data


def get_redata(path, file):
    mat = loadmat(os.path.join(path, file))
    df_len = np.min([len(mat['emg']), len(mat['restimulus'])])
    data = pd.DataFrame(mat['emg'][:df_len])
    data['restimulus'] = mat['restimulus'][:df_len]
    data['rerepetition'] = mat['rerepetition'][:df_len]

    return data


def normalise(data, train_reps, channel=12):
    x = [np.where(data.values[:, channel + 1] == rep) for rep in train_reps]
    indices = np.squeeze(np.concatenate(x, axis=-1))
    train_data = data.iloc[indices, :]
    train_data = data.reset_index(drop=True)
    scaler = StandardScaler(with_mean=True,
                            with_std=True,
                            copy=False).fit(train_data.iloc[:, :channel])

    scaled = scaler.transform(data.iloc[:, :channel])
    normalised = pd.DataFrame(scaled)
    normalised['stimulus'] = data['stimulus'].values
    normalised['repetition'] = data['repetition'].values

    return normalised


def filter_data(data, f, butterworth_order=4, btype='lowpass', channel=12):
    emg_data = data.values[:, :channel]

    f_sampling = 2000
    nyquist = f_sampling / 2
    if isinstance(f, int):
        fc = f / nyquist
    else:
        fc = list(f)
        for i in range(len(f)):
            fc[i] = fc[i] / nyquist

    b, a = signal.butter(butterworth_order, fc, btype=btype)
    transpose = emg_data.T.copy()

    for i in range(len(transpose)):
        transpose[i] = (signal.lfilter(b, a, transpose[i]))

    filtered = pd.DataFrame(transpose.T)
    filtered['stimulus'] = data['stimulus'].values
    filtered['repetition'] = data['repetition'].values

    return filtered


def refilter_data(data, f=(10, 90), butterworth_order=4, btype='bandpass', channel=16):
    """
    对EMG信号进行带通滤波，只处理EMG，不处理标签。
    """
    if hasattr(data, 'values'):
        emg_data = data.values[:, :channel]
    else:
        emg_data = data[:, :channel] if data.ndim == 2 else data
    # 滤波处理
    f_sampling = 2000
    nyquist = f_sampling / 2
    if isinstance(f, int):
        fc = f / nyquist
    else:
        fc = list(f)
        for i in range(len(f)):
            fc[i] = fc[i] / nyquist

    b, a = signal.butter(butterworth_order, fc, btype=btype)
    transpose = emg_data.T.copy()

    for i in range(len(transpose)):
        transpose[i] = (signal.lfilter(b, a, transpose[i]))

    filtered = pd.DataFrame(transpose.T)
    return filtered  # 只返回滤波后的EMG信号


def rectify(data):
    return abs(data)


def notch_refilter(data, f0=50, Q=30, fs=200, channel=16):
    """
    对EMG信号进行陷波滤波，只处理EMG，不处理标签。
    """
    if hasattr(data, 'values'):
        emg_data = data.values[:, :channel]
    else:
        emg_data = data[:, :channel] if data.ndim == 2 else data
    # 陷波处理
    nyquist = fs / 2
    f0 = f0 / nyquist

    b, a = signal.iirnotch(f0, Q)

    transpose = emg_data.T.copy()
    for i in range(len(transpose)):
        transpose[i] = signal.filtfilt(b, a, transpose[i])

    filtered = pd.DataFrame(transpose.T)
    return filtered  # 只返回滤波后的EMG信号


def windowing(data, reps, gestures, win_len, win_stride):
    if reps:
        x = [np.where(data.values[:, 13] == rep) for rep in reps]
        indices = np.squeeze(np.concatenate(x, axis=-1))
        data = data.iloc[indices, :]
        data = data.reset_index(drop=True)

    if gestures:
        x = [np.where(data.values[:, 12] == move) for move in gestures]
        indices = np.squeeze(np.concatenate(x, axis=-1))
        data = data.iloc[indices, :]
        data = data.reset_index(drop=True)

    # 没有reps和gestures标签时，保留所有数据
    idx = [i for i in range(win_len, len(data), win_stride)]

    X = np.zeros([len(idx), win_len, len(data.columns) - 2])
    y = np.zeros([len(idx), ])
    reps = np.zeros([len(idx), ])

    for i, end in enumerate(tqdm(idx)):
        start = end - win_len
        X[i] = data.iloc[start:end, 0:12]
        y[i] = data.iloc[end, 12]
        reps[i] = data.iloc[end, 13]

    return X, y, reps


# 注释掉TensorFlow相关的训练函数，改用纯Python实现
# def train_model(model, X_train_wind, y_train_wind, X_test_wind, y_test_wind, save_to, epoch=300):
#     from tensorflow import keras as K
#     opt_adam = K.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#     model.compile(loss='categorical_crossentropy', optimizer=opt_adam, metrics=['categorical_accuracy'])
# 
#     #         log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# 
#     es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
# 
#     # tb = TensorBoard(log_dir=log_dir, histogram_freq=1)
# 
#     mc = ModelCheckpoint(save_to, monitor='val_categorical_accuracy', mode='max', save_best_only=True)
#     # mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', save_best_only=True)
#     # Train the model
#     history = model.fit(
#         X_train_wind, K.utils.to_categorical(y_train_wind),
#         batch_size=256,
#         epochs=epoch,
#         verbose=1,
#         validation_data=(X_test_wind, K.utils.to_categorical(y_test_wind)),
#         callbacks=[es, mc],  # , tb])
#     )
#     # Evaluate the model
#     score = model.evaluate(X_test_wind, K.utils.to_categorical(y_test_wind), verbose=0)
#     return model, history.history, score


# 注释掉TensorFlow相关的模型加载函数
# def load_model_weights(model, weights_path):
#     if os.path.exists(weights_path):
#         model.load_weights(weights_path)
#         print("加载模型权重成功")
#     else:
#         print("模型权重文件不存在")
#     return model


def divide_data(emg_data, label_data, repetition_data, train_reps, test_reps):
    train_index = np.where(np.isin(repetition_data, train_reps))[0]
    test_index = np.where(np.isin(repetition_data, test_reps))[0]

    emg_train, emg_test = emg_data[train_index], emg_data[test_index]
    label_train, label_test = label_data[train_index], label_data[test_index]
    rep_train, rep_test = repetition_data[train_index], repetition_data[test_index]

    return emg_train, emg_test, label_train, label_test, rep_train, rep_test


# 注释掉TensorFlow相关的测试函数
# def test_model(model, X_test, y_test):
#     y_pred = np.argmax(model.predict(X_test), axis=1)
#     y_true = np.argmax(y_test, axis=1)
#     cm = confusion_matrix(y_true, y_pred)
#     accuracy = np.sum(y_pred == y_true) / len(y_true)
#     print("测试准确率：", accuracy)
#     print("混淆矩阵：\n", cm)
#     return accuracy, cm


# 注释掉TensorFlow相关的预测函数
# def predict_gestures(model, emg_data):
#     # 对新的emg数据进行手势预测
#     emg_data = np.expand_dims(emg_data, axis=0)
#     prediction = model.predict(emg_data)
#     predicted_gesture = np.argmax(prediction, axis=1)[0]
#     return predicted_gesture

def notch_filter(data, f0, Q, fs=2000, channel=12):
    # 陷波滤波
    nyquist = fs / 2
    f0 = f0 / nyquist
    b, a = signal.iirnotch(f0, Q)
    y = signal.filtfilt(b, a, data)
    return y

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    # 带通滤波
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.filtfilt(b, a, data)
    return y



def feature_combine(feature_list, axis=-1):
    # 特征组合
    combined_feature = np.concatenate(feature_list, axis=axis)
    return combined_feature

def get_categorical(y):
    import pandas as pd
    return pd.get_dummies(pd.Series(y)).values

# ===================== 改进的EMG滤波方案 =====================

def advanced_emg_filter(x, fs=2000, region='EU'):
    """
    改进的EMG信号滤波方案
    顺序：高通20Hz → 陷波50Hz → 陷波100Hz → 低通450Hz
    
    Parameters:
    -----------
    x : numpy.ndarray
        输入EMG信号，形状为 (samples, channels) 或 (samples,)
    fs : int
        采样频率，默认2000Hz
    region : str
        地区设置，'EU'为50/100Hz工频，'US'为60/120Hz工频
    
    Returns:
    --------
    filtered_x : numpy.ndarray
        滤波后的EMG信号，与输入形状相同
    """
    # 确保输入是numpy数组
    x = np.asarray(x, dtype=np.float32)
    original_shape = x.shape
    
    # 如果是1D数组，转换为2D
    if x.ndim == 1:
        x = x.reshape(-1, 1)
        was_1d = True
    else:
        was_1d = False
    
    # 根据地区选择工频
    if region == 'EU':
        notch_freqs = [50, 100]
    elif region == 'US':
        notch_freqs = [60, 120]
    else:
        notch_freqs = [50, 100]  # 默认欧洲标准
    
    try:
        # 1. 高通滤波器 (20Hz)
        sos_hp = signal.butter(4, 20, btype='highpass', fs=fs, output='sos')
        
        # 2. 陷波滤波器 (50Hz, 100Hz)
        sos_notches = []
        for f0 in notch_freqs:
            b, a = signal.iirnotch(f0, Q=30, fs=fs)
            sos_notches.append(signal.tf2sos(b, a))
        
        # 3. 低通滤波器 (450Hz)
        sos_lp = signal.butter(4, 450, btype='lowpass', fs=fs, output='sos')
        
        # 4. 串联滤波
        for channel in range(x.shape[1]):
            channel_data = x[:, channel]
            
            # 应用高通滤波
            channel_data = signal.sosfiltfilt(sos_hp, channel_data)
            
            # 应用陷波滤波
            for sos_notch in sos_notches:
                channel_data = signal.sosfiltfilt(sos_notch, channel_data)
            
            # 应用低通滤波
            channel_data = signal.sosfiltfilt(sos_lp, channel_data)
            
            x[:, channel] = channel_data
        
        # 恢复原始形状
        if was_1d:
            x = x.flatten()
        
        return x.astype(np.float32)
        
    except Exception as e:
        print(f"滤波过程中出现错误: {e}")
        print("使用原始信号作为备选")
        return np.asarray(x, dtype=np.float32)

def check_filter_quality(raw, filtered, fs=2000, channel_idx=0):
    """
    检查滤波质量
    
    Parameters:
    -----------
    raw : numpy.ndarray
        原始EMG信号
    filtered : numpy.ndarray
        滤波后的EMG信号
    fs : int
        采样频率
    channel_idx : int
        用于分析的通道索引
    
    Returns:
    --------
    quality_ok : bool
        滤波质量是否合格
    """
    try:
        # 确保输入是numpy数组
        raw = np.asarray(raw)
        filtered = np.asarray(filtered)
        
        # 如果是2D数组，选择指定通道
        if raw.ndim == 2:
            raw_channel = raw[:, channel_idx]
            filtered_channel = filtered[:, channel_idx]
        else:
            raw_channel = raw
            filtered_channel = filtered
        
        # 计算功率谱
        nperseg = min(4096, len(raw_channel)//4)
        if nperseg < 64:  # 确保nperseg足够大
            nperseg = 64
        if nperseg > len(raw_channel)//2:  # 确保nperseg不会太大
            nperseg = len(raw_channel)//2
        if nperseg < 32:  # 如果还是太小，设为最小值
            nperseg = 32
            
        f, Praw = signal.welch(raw_channel, fs, nperseg=nperseg)
        _, Pfilt = signal.welch(filtered_channel, fs, nperseg=nperseg)
        
        # 检查低频抑制 (<20Hz)
        low_freq_mask = f < 20
        if np.any(low_freq_mask):
            low_freq_attn = 10 * np.log10(np.mean(Pfilt[low_freq_mask]) / np.mean(Praw[low_freq_mask]))
        else:
            low_freq_attn = -100  # 如果没有低频成分，设为-100dB
        
        # 检查工频抑制 (50Hz附近)
        notch_50_mask = (f >= 48) & (f <= 52)
        if np.any(notch_50_mask):
            notch_50_attn = 10 * np.log10(np.mean(Pfilt[notch_50_mask]) / np.mean(Praw[notch_50_mask]))
        else:
            notch_50_attn = -100
        
        # 检查工频抑制 (100Hz附近)
        notch_100_mask = (f >= 98) & (f <= 102)
        if np.any(notch_100_mask):
            notch_100_attn = 10 * np.log10(np.mean(Pfilt[notch_100_mask]) / np.mean(Praw[notch_100_mask]))
        else:
            notch_100_attn = -100
        
        # 打印结果
        print(f"滤波质量检查结果:")
        print(f"  低频抑制 (<20Hz): {low_freq_attn:.1f} dB")
        print(f"  50Hz抑制: {notch_50_attn:.1f} dB")
        print(f"  100Hz抑制: {notch_100_attn:.1f} dB")
        
        # 判断质量是否合格
        quality_ok = (low_freq_attn < -20 and 
                     notch_50_attn < -30 and 
                     notch_100_attn < -30)
        
        if quality_ok:
            print("  ✓ 滤波质量合格")
        else:
            print("  ✗ 滤波质量不合格")
        
        return quality_ok
        
    except Exception as e:
        print(f"质量检查过程中出现错误: {e}")
        return False

def adaptive_emg_filter(x, fs=2000, region='EU', check_quality=False):
    """
    自适应EMG滤波，支持质量检查
    
    Parameters:
    -----------
    x : numpy.ndarray
        输入EMG信号
    fs : int
        采样频率
    region : str
        地区设置
    check_quality : bool
        是否进行质量检查
    
    Returns:
    --------
    filtered_x : numpy.ndarray
        滤波后的EMG信号
    """
    # 应用滤波
    filtered_x = advanced_emg_filter(x, fs, region)
    
    # 可选的质量检查
    if check_quality:
        quality_ok = check_filter_quality(x, filtered_x, fs)
        if not quality_ok:
            print("警告: 滤波质量不合格，请检查参数设置")
    
    return filtered_x
