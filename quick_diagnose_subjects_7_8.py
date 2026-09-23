#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速诊断受试者7和8的原始信号问题
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def quick_signal_check(data_dir, subject_id, exercise=1):
    """快速检查信号质量"""
    file_path = os.path.join(data_dir, f's{subject_id}', f'S{subject_id}_E{exercise}_A1.mat')
    
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return None
    
    try:
        mat = scio.loadmat(file_path)
        emg = mat['emg']
        stimulus = mat['stimulus'].flatten()
        
        print(f"受试者{subject_id} Exercise{exercise}:")
        print(f"  数据形状: {emg.shape}")
        print(f"  采样点数: {len(emg)}")
        print(f"  通道数: {emg.shape[1]}")
        
        # 计算基本统计
        rms_values = np.sqrt(np.mean(emg**2, axis=0))
        std_values = np.std(emg, axis=0)
        peak_to_peak = np.max(emg, axis=0) - np.min(emg, axis=0)
        
        print(f"  平均RMS: {np.mean(rms_values):.2f}")
        print(f"  平均标准差: {np.mean(std_values):.2f}")
        print(f"  平均峰峰值: {np.mean(peak_to_peak):.2f}")
        
        return {
            'emg': emg,
            'stimulus': stimulus,
            'rms': rms_values,
            'std': std_values,
            'peak_to_peak': peak_to_peak
        }
        
    except Exception as e:
        print(f"加载失败: {e}")
        return None

def plot_signal_samples(data, subject_id, exercise, save_path=None):
    """绘制信号样本"""
    emg = data['emg']
    stimulus = data['stimulus']
    
    # 找到手势区域
    gesture_starts = np.where(np.diff(stimulus) > 0)[0]
    gesture_ends = np.where(np.diff(stimulus) < 0)[0]
    
    # 选择前8个通道进行可视化
    channels_to_plot = min(8, emg.shape[1])
    
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    fig.suptitle(f'受试者{subject_id} Exercise{exercise} 原始EMG信号样本', fontsize=16)
    
    for ch in range(channels_to_plot):
        row = ch // 2
        col = ch % 2
        
        # 绘制完整信号
        time_axis = np.arange(len(emg[:, ch])) / 2000  # 2000Hz采样率
        axes[row, col].plot(time_axis, emg[:, ch], linewidth=0.5, alpha=0.7)
        
        # 标记手势区域
        for start, end in zip(gesture_starts, gesture_ends):
            if end > start:
                axes[row, col].axvspan(start/2000, end/2000, alpha=0.2, color='red')
        
        # 设置标题和标签
        rms_val = data['rms'][ch]
        std_val = data['std'][ch]
        axes[row, col].set_title(f'通道{ch+1}\nRMS: {rms_val:.1f}, STD: {std_val:.1f}')
        axes[row, col].set_xlabel('时间 (s)')
        axes[row, col].set_ylabel('幅值 (μV)')
        axes[row, col].grid(True, alpha=0.3)
        
        # 设置y轴范围
        y_range = np.percentile(emg[:, ch], [1, 99])
        axes[row, col].set_ylim(y_range[0] * 1.1, y_range[1] * 1.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def compare_subjects_metrics(data_dir):
    """比较两个受试者的指标"""
    subjects = [7, 8]
    exercises = [1, 2, 3]
    
    all_metrics = {}
    
    for subject_id in subjects:
        print(f"\n{'='*50}")
        print(f"分析受试者 {subject_id}")
        print(f"{'='*50}")
        
        subject_metrics = {}
        
        for exercise in exercises:
            print(f"\n--- Exercise {exercise} ---")
            data = quick_signal_check(data_dir, subject_id, exercise)
            
            if data is not None:
                subject_metrics[exercise] = data
                
                # 绘制信号
                plot_signal_samples(data, subject_id, exercise, 
                                  f'subject_{subject_id}_exercise_{exercise}_signals.png')
        
        all_metrics[subject_id] = subject_metrics
    
    # 创建比较图表
    create_comparison_plot(all_metrics)
    
    return all_metrics

def create_comparison_plot(all_metrics):
    """创建比较图表"""
    subjects = list(all_metrics.keys())
    exercises = [1, 2, 3]
    
    # 准备数据
    rms_data = {subject: [] for subject in subjects}
    std_data = {subject: [] for subject in subjects}
    
    for subject in subjects:
        for exercise in exercises:
            if exercise in all_metrics[subject]:
                avg_rms = np.mean(all_metrics[subject][exercise]['rms'])
                avg_std = np.mean(all_metrics[subject][exercise]['std'])
                rms_data[subject].append(avg_rms)
                std_data[subject].append(avg_std)
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    x = np.arange(len(exercises))
    width = 0.35
    
    # RMS比较
    ax1.bar(x - width/2, rms_data[subjects[0]], width, label=f'受试者{subjects[0]}', alpha=0.8)
    ax1.bar(x + width/2, rms_data[subjects[1]], width, label=f'受试者{subjects[1]}', alpha=0.8)
    ax1.set_xlabel('Exercise')
    ax1.set_ylabel('平均RMS值')
    ax1.set_title('信号强度比较')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'E{ex}' for ex in exercises])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 标准差比较
    ax2.bar(x - width/2, std_data[subjects[0]], width, label=f'受试者{subjects[0]}', alpha=0.8)
    ax2.bar(x + width/2, std_data[subjects[1]], width, label=f'受试者{subjects[1]}', alpha=0.8)
    ax2.set_xlabel('Exercise')
    ax2.set_ylabel('平均标准差')
    ax2.set_title('信号变异性比较')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'E{ex}' for ex in exercises])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('subjects_7_8_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def diagnose_problems(all_metrics):
    """诊断问题"""
    print(f"\n{'='*60}")
    print("问题诊断报告")
    print(f"{'='*60}")
    
    for subject_id in all_metrics:
        print(f"\n受试者 {subject_id} 问题分析:")
        
        for exercise in all_metrics[subject_id]:
            data = all_metrics[subject_id][exercise]
            print(f"\n  Exercise {exercise}:")
            
            # 检查每个通道
            for ch in range(len(data['rms'])):
                rms = data['rms'][ch]
                std = data['std'][ch]
                peak_to_peak = data['peak_to_peak'][ch]
                
                problems = []
                
                # 判断标准（这些阈值可能需要根据实际情况调整）
                if rms < 30:  # 信号太弱
                    problems.append("信号强度过低")
                if std < 10:  # 信号变化太小
                    problems.append("信号变化过小")
                if peak_to_peak < 50:  # 峰峰值太小
                    problems.append("峰峰值过小")
                if std > 100:  # 噪声太大
                    problems.append("噪声水平过高")
                
                if problems:
                    print(f"    通道{ch+1}: {', '.join(problems)}")
                    print(f"      RMS: {rms:.1f}, STD: {std:.1f}, 峰峰值: {peak_to_peak:.1f}")

def main():
    """主函数"""
    print("=== 受试者7和8快速诊断 ===")
    
    # 检查数据目录
    data_dir = 'D:/DB4'
    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        print("请检查数据路径或修改脚本中的data_dir变量")
        return
    
    # 执行诊断
    all_metrics = compare_subjects_metrics(data_dir)
    
    # 生成问题诊断报告
    diagnose_problems(all_metrics)
    
    print(f"\n{'='*60}")
    print("诊断完成！")
    print("生成的文件:")
    print("- subject_*_exercise_*_signals.png: 各受试者各Exercise的信号图")
    print("- subjects_7_8_comparison.png: 两个受试者的比较图")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 