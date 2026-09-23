import math
import h5py
import pandas as pd
import numpy as np



def get_threeSet(emg, label, rep_arr, rep_vali):
    train_reps = [1, 3, 4, 6]
    test_reps = [5]  # 只使用重复次数5作为测试集
    vali_reps = [rep_vali]  # 使用指定的重复次数作为验证集
    
    # 从训练集中移除验证集重复次数
    if rep_vali in train_reps:
        train_reps.remove(rep_vali)
    
    # 获取测试集数据
    x = [np.where(rep_arr == rep)[0] for rep in test_reps]  # 添加[0]来获取索引数组
    indices = np.concatenate(x, axis=0)
    # 边界检查
    indices = indices[indices < len(emg)]
    emg_test = emg[indices, :]
    label_test = label[indices]
    rep_test = rep_arr[indices]
    
    # 获取验证集数据
    x = [np.where(rep_arr == rep)[0] for rep in vali_reps]  # 添加[0]来获取索引数组
    indices2 = np.concatenate(x, axis=0)
    # 边界检查
    indices2 = indices2[indices2 < len(emg)]
    emg_vali = emg[indices2, :]
    label_vali = label[indices2]
    rep_vali = rep_arr[indices2]
    
    # 获取训练集数据
    x = [np.where(rep_arr == rep)[0] for rep in train_reps]  # 添加[0]来获取索引数组
    indices3 = np.concatenate(x, axis=0)
    # 边界检查
    indices3 = indices3[indices3 < len(emg)]
    emg_train = emg[indices3, :]
    label_train = label[indices3]
    rep_train = rep_arr[indices3]

    return emg_train, emg_vali, emg_test, label_train, label_vali, label_test


def get_threeSet_indices(rep_arr, rep_vali=2):
    """
    返回训练、验证、测试集的索引
    Args:
        rep_arr: 重复次数数组
        rep_vali: 验证集使用的重复次数，默认2
    Returns:
        train_indices, val_indices, test_indices: 三个索引数组
    """
    train_reps = [1, 3, 4, 6]
    test_reps = [5]  # 只使用重复次数5作为测试集
    vali_reps = [rep_vali]  # 使用指定的重复次数作为验证集
    
    # 从训练集中移除验证集重复次数
    if rep_vali in train_reps:
        train_reps.remove(rep_vali)
    
    # 获取测试集索引
    x = [np.where(rep_arr == rep)[0] for rep in test_reps]
    test_indices = np.concatenate(x, axis=0)
    
    # 获取验证集索引
    x = [np.where(rep_arr == rep)[0] for rep in vali_reps]
    val_indices = np.concatenate(x, axis=0)
    
    # 获取训练集索引
    x = [np.where(rep_arr == rep)[0] for rep in train_reps]
    train_indices = np.concatenate(x, axis=0)

    return train_indices, val_indices, test_indices


def get_twoSet(emg, label, rep_arr):
    train_reps = [1, 3, 4, 6]
    test_reps = [5]  # 只使用重复次数5作为测试集，避免与验证集重叠
    x = [np.where(rep_arr == rep)[0] for rep in test_reps]
    indices = np.concatenate(x, axis=0)
    # 边界检查
    indices = indices[indices < len(emg)]
    emg_test = emg[indices, :]
    label_test = label[indices]
    x = [np.where(rep_arr == rep)[0] for rep in train_reps]
    indices3 = np.concatenate(x, axis=0)
    # 边界检查
    indices3 = indices3[indices3 < len(emg)]
    emg_train = emg[indices3, :]
    label_train = label[indices3]

    return emg_train, emg_test, label_train, label_test

# if __name__ == '__main__':
#     train_reps = [1, 3, 4, 6]
#     test_reps = 3
#     train_reps.remove(test_reps)
#     train_reps
