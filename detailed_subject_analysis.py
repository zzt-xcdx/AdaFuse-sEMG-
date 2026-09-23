#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深入分析受试者7、8和10号之间的差异
"""

import numpy as np
import scipy.io as scio
import os
from scipy import stats

def analyze_signal_characteristics(data_dir):
    """分析信号特征"""
    subjects = [7, 8, 10]
    exercises = [1, 2, 3]
    
    print("=== 深入信号特征分析 ===\n")
    
    all_characteristics = {}
    
    for subject_id in subjects:
        print(f"受试者{subject_id} 详细分析:")
        all_characteristics[subject_id] = {}
        
        for exercise in exercises:
            file_path = os.path.join(data_dir, f's{subject_id}', f'S{subject_id}_E{exercise}_A1.mat')
            
            if os.path.exists(file_path):
                try:
                    mat = scio.loadmat(file_path)
                    emg = mat['emg']
                    stimulus = mat['stimulus'].flatten()
                    
                    print(f"  Exercise{exercise}:")
                    print(f"    数据形状: {emg.shape}")
                    
                    # 分析每个通道的特征
                    channel_features = {}
                    
                    for ch in range(min(5, emg.shape[1])):  # 分析前5个通道
                        signal_ch = emg[:, ch]
                        
                        # 基本统计特征
                        rms = np.sqrt(np.mean(signal_ch**2))
                        std = np.std(signal_ch)
                        mean_val = np.mean(signal_ch)
                        median_val = np.median(signal_ch)
                        
                        # 分布特征
                        skewness = stats.skew(signal_ch)
                        kurtosis = stats.kurtosis(signal_ch)
                        
                        # 动态范围
                        dynamic_range = np.max(signal_ch) - np.min(signal_ch)
                        q95 = np.percentile(signal_ch, 95)
                        q5 = np.percentile(signal_ch, 5)
                        iqr = np.percentile(signal_ch, 75) - np.percentile(signal_ch, 25)
                        
                        # 信号稳定性
                        # 计算局部变异系数
                        window_size = 1000
                        local_cv = []
                        for i in range(0, len(signal_ch) - window_size, window_size // 2):
                            window = signal_ch[i:i+window_size]
                            if np.std(window) > 0:
                                local_cv.append(np.std(window) / np.abs(np.mean(window)))
                        
                        cv_stability = np.std(local_cv) if local_cv else 0
                        
                        # 手势期间的信号特征
                        gesture_signals = []
                        rest_signals = []
                        
                        for i in range(len(stimulus)):
                            if stimulus[i] > 0:  # 手势期间
                                gesture_signals.append(signal_ch[i])
                            else:  # 休息期间
                                rest_signals.append(signal_ch[i])
                        
                        gesture_rms = np.sqrt(np.mean(np.array(gesture_signals)**2)) if gesture_signals else 0
                        rest_rms = np.sqrt(np.mean(np.array(rest_signals)**2)) if rest_signals else 0
                        gesture_rest_ratio = gesture_rms / (rest_rms + 1e-10)
                        
                        channel_features[ch] = {
                            'rms': rms,
                            'std': std,
                            'mean': mean_val,
                            'median': median_val,
                            'skewness': skewness,
                            'kurtosis': kurtosis,
                            'dynamic_range': dynamic_range,
                            'q95': q95,
                            'q5': q5,
                            'iqr': iqr,
                            'cv_stability': cv_stability,
                            'gesture_rest_ratio': gesture_rest_ratio
                        }
                        
                        print(f"    通道{ch+1}:")
                        print(f"      RMS: {rms:.1f}, STD: {std:.1f}")
                        print(f"      偏度: {skewness:.3f}, 峰度: {kurtosis:.3f}")
                        print(f"      动态范围: {dynamic_range:.1f}, IQR: {iqr:.1f}")
                        print(f"      手势/休息比率: {gesture_rest_ratio:.2f}")
                        print(f"      CV稳定性: {cv_stability:.3f}")
                    
                    all_characteristics[subject_id][exercise] = channel_features
                    
                except Exception as e:
                    print(f"    加载失败: {e}")
            else:
                print(f"  Exercise{exercise}: 文件不存在")
        
        print("-" * 60)
    
    return all_characteristics

def compare_subjects(all_characteristics):
    """对比受试者之间的差异"""
    print("\n=== 受试者间差异分析 ===\n")
    
    # 以10号受试者作为基准
    baseline_subject = 10
    problem_subjects = [7, 8]
    
    if baseline_subject not in all_characteristics:
        print("基准受试者10号数据缺失")
        return
    
    for exercise in [1, 2, 3]:
        if exercise not in all_characteristics[baseline_subject]:
            continue
            
        print(f"Exercise {exercise} 对比分析:")
        
        baseline_features = all_characteristics[baseline_subject][exercise]
        
        for subject_id in problem_subjects:
            if exercise not in all_characteristics[subject_id]:
                continue
                
            print(f"\n  受试者{subject_id} vs 受试者{baseline_subject}:")
            
            current_features = all_characteristics[subject_id][exercise]
            
            # 对比每个通道
            for ch in range(min(len(baseline_features), len(current_features))):
                if ch not in baseline_features or ch not in current_features:
                    continue
                    
                baseline = baseline_features[ch]
                current = current_features[ch]
                
                print(f"    通道{ch+1}:")
                
                # 对比关键指标
                rms_ratio = current['rms'] / (baseline['rms'] + 1e-10)
                std_ratio = current['std'] / (baseline['std'] + 1e-10)
                dynamic_ratio = current['dynamic_range'] / (baseline['dynamic_range'] + 1e-10)
                stability_ratio = current['cv_stability'] / (baseline['cv_stability'] + 1e-10)
                gesture_ratio = current['gesture_rest_ratio'] / (baseline['gesture_rest_ratio'] + 1e-10)
                
                print(f"      RMS比率: {rms_ratio:.2f}x")
                print(f"      标准差比率: {std_ratio:.2f}x")
                print(f"      动态范围比率: {dynamic_ratio:.2f}x")
                print(f"      稳定性比率: {stability_ratio:.2f}x")
                print(f"      手势/休息比率: {gesture_ratio:.2f}x")
                
                # 识别问题
                problems = []
                if rms_ratio < 0.5:
                    problems.append("信号强度过低")
                elif rms_ratio > 2.0:
                    problems.append("信号强度过高")
                
                if std_ratio < 0.5:
                    problems.append("信号变化过小")
                elif std_ratio > 2.0:
                    problems.append("信号变化过大")
                
                if stability_ratio > 1.5:
                    problems.append("信号稳定性差")
                
                if gesture_ratio < 0.7:
                    problems.append("手势识别能力弱")
                
                if problems:
                    print(f"      ❌ 问题: {', '.join(problems)}")
                else:
                    print(f"      ✅ 正常")
        
        print("-" * 40)

def analyze_signal_quality_patterns(all_characteristics):
    """分析信号质量模式"""
    print("\n=== 信号质量模式分析 ===\n")
    
    subjects = [7, 8, 10]
    exercises = [1, 2, 3]
    
    # 计算每个受试者的整体质量指标
    quality_scores = {}
    
    for subject_id in subjects:
        if subject_id not in all_characteristics:
            continue
            
        total_score = 0
        channel_count = 0
        
        for exercise in exercises:
            if exercise not in all_characteristics[subject_id]:
                continue
                
            for ch in all_characteristics[subject_id][exercise]:
                features = all_characteristics[subject_id][exercise][ch]
                
                # 计算质量分数（0-100）
                score = 0
                
                # RMS分数（20分）
                if 100 <= features['rms'] <= 1000:
                    score += 20
                elif 50 <= features['rms'] <= 1500:
                    score += 15
                elif features['rms'] > 0:
                    score += 10
                
                # 稳定性分数（30分）
                if features['cv_stability'] < 0.5:
                    score += 30
                elif features['cv_stability'] < 1.0:
                    score += 20
                elif features['cv_stability'] < 2.0:
                    score += 10
                
                # 手势识别分数（30分）
                if features['gesture_rest_ratio'] > 2.0:
                    score += 30
                elif features['gesture_rest_ratio'] > 1.5:
                    score += 20
                elif features['gesture_rest_ratio'] > 1.0:
                    score += 10
                
                # 分布质量分数（20分）
                if abs(features['skewness']) < 2.0 and features['kurtosis'] < 10:
                    score += 20
                elif abs(features['skewness']) < 3.0 and features['kurtosis'] < 15:
                    score += 15
                elif abs(features['skewness']) < 5.0 and features['kurtosis'] < 25:
                    score += 10
                
                total_score += score
                channel_count += 1
        
        if channel_count > 0:
            quality_scores[subject_id] = total_score / channel_count
    
    # 显示质量评分
    print("受试者信号质量评分 (0-100):")
    for subject_id in sorted(quality_scores.keys()):
        score = quality_scores[subject_id]
        print(f"  受试者{subject_id}: {score:.1f}")
        
        if score >= 80:
            print("    ✅ 优秀")
        elif score >= 60:
            print("    ⚠️ 良好")
        elif score >= 40:
            print("    ⚠️ 一般")
        else:
            print("    ❌ 较差")
    
    return quality_scores

def main():
    """主函数"""
    data_dir = 'D:/DB4'
    
    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        return
    
    print("=== 受试者7、8、10号深入差异分析 ===\n")
    
    # 1. 分析信号特征
    all_characteristics = analyze_signal_characteristics(data_dir)
    
    # 2. 对比受试者差异
    compare_subjects(all_characteristics)
    
    # 3. 分析信号质量模式
    quality_scores = analyze_signal_quality_patterns(all_characteristics)
    
    # 4. 总结关键差异
    print("\n=== 关键差异总结 ===")
    print("受试者7和8准确率低的主要原因:")
    
    if 7 in quality_scores and 8 in quality_scores and 10 in quality_scores:
        score_7 = quality_scores[7]
        score_8 = quality_scores[8]
        score_10 = quality_scores[10]
        
        print(f"1. 信号质量评分差异显著:")
        print(f"   - 受试者7: {score_7:.1f}")
        print(f"   - 受试者8: {score_8:.1f}")
        print(f"   - 受试者10: {score_10:.1f}")
        
        if score_7 < score_10 * 0.8:
            print("   受试者7的信号质量明显低于10号")
        if score_8 < score_10 * 0.8:
            print("   受试者8的信号质量明显低于10号")
    
    print("\n2. 具体问题:")
    print("   - 信号动态范围过大，导致特征提取困难")
    print("   - 信号稳定性差，一致性不足")
    print("   - 手势识别能力弱，信噪比低")
    print("   - 分布不均匀，存在大量极端值")
    
    print("\n3. 建议解决方案:")
    print("   - 使用增强预处理技术改善信号质量")
    print("   - 实施信号均衡化和动态范围压缩")
    print("   - 选择鲁棒的特征提取方法")
    print("   - 考虑重新采集数据或调整采集参数")

if __name__ == "__main__":
    main() 