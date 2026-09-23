#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
鲁棒特征选择模块
选择最稳定的特征子集，提高模型对不稳定信号的鲁棒性
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import h5py
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class RobustFeatureSelector:
    """鲁棒特征选择器"""
    
    def __init__(self, n_features=50):
        self.n_features = n_features
        self.selected_features = None
        self.feature_scores = None
        self.scaler = StandardScaler()
        
    def calculate_feature_stability(self, features, labels):
        """计算特征稳定性"""
        stability_scores = []
        
        for i in range(features.shape[1]):
            feature_values = features[:, i]
            
            # 计算每个手势类别的特征稳定性
            gesture_stabilities = []
            unique_labels = np.unique(labels)
            
            for label in unique_labels:
                mask = labels == label
                if np.sum(mask) > 10:  # 至少需要10个样本
                    gesture_feature = feature_values[mask]
                    
                    # 计算变异系数（越小越稳定）
                    mean_val = np.mean(gesture_feature)
                    std_val = np.std(gesture_feature)
                    cv = std_val / np.abs(mean_val) if mean_val != 0 else 0
                    
                    # 稳定性评分（变异系数越小，稳定性越高）
                    stability = 1.0 / (1.0 + cv)
                    gesture_stabilities.append(stability)
            
            # 平均稳定性
            avg_stability = np.mean(gesture_stabilities) if gesture_stabilities else 0
            stability_scores.append(avg_stability)
        
        return np.array(stability_scores)
    
    def calculate_feature_discriminability(self, features, labels):
        """计算特征区分度"""
        # 使用F统计量
        f_scores, _ = f_classif(features, labels)
        
        # 使用互信息
        mi_scores = mutual_info_classif(features, labels, random_state=42)
        
        # 综合评分
        discriminability_scores = (f_scores + mi_scores) / 2
        
        return discriminability_scores
    
    def calculate_feature_redundancy(self, features):
        """计算特征冗余度"""
        n_features = features.shape[1]
        redundancy_scores = np.zeros(n_features)
        
        # 计算特征间的相关性
        corr_matrix = np.corrcoef(features.T)
        
        for i in range(n_features):
            # 计算与其他特征的平均相关性
            correlations = np.abs(corr_matrix[i, :])
            correlations[i] = 0  # 排除自身
            avg_correlation = np.mean(correlations)
            
            # 冗余度评分（相关性越低，冗余度越小）
            redundancy_scores[i] = 1.0 - avg_correlation
        
        return redundancy_scores
    
    def calculate_feature_robustness(self, features, labels):
        """计算特征鲁棒性"""
        robustness_scores = []
        
        for i in range(features.shape[1]):
            feature_values = features[:, i]
            
            # 1. 对噪声的鲁棒性（添加少量噪声后特征变化）
            noise_level = 0.01 * np.std(feature_values)
            noisy_features = feature_values + np.random.normal(0, noise_level, len(feature_values))
            
            # 计算原始特征和噪声特征的相关系数
            correlation = np.corrcoef(feature_values, noisy_features)[0, 1]
            noise_robustness = correlation if not np.isnan(correlation) else 0
            
            # 2. 对异常值的鲁棒性
            # 使用中位数而不是均值
            median_val = np.median(feature_values)
            mean_val = np.mean(feature_values)
            outlier_robustness = 1.0 - np.abs(median_val - mean_val) / (np.abs(mean_val) + 1e-8)
            
            # 3. 特征分布的一致性
            # 计算偏度和峰度
            skewness = np.mean(((feature_values - np.mean(feature_values)) / np.std(feature_values)) ** 3)
            kurtosis = np.mean(((feature_values - np.mean(feature_values)) / np.std(feature_values)) ** 4) - 3
            
            # 分布一致性（越接近正态分布越好）
            distribution_consistency = 1.0 / (1.0 + np.abs(skewness) + np.abs(kurtosis))
            
            # 综合鲁棒性评分
            robustness = (noise_robustness + outlier_robustness + distribution_consistency) / 3
            robustness_scores.append(robustness)
        
        return np.array(robustness_scores)
    
    def select_features(self, features, labels):
        """选择最优特征子集"""
        print("开始特征选择...")
        
        # 标准化特征
        features_scaled = self.scaler.fit_transform(features)
        
        # 计算各种评分
        print("计算特征稳定性...")
        stability_scores = self.calculate_feature_stability(features_scaled, labels)
        
        print("计算特征区分度...")
        discriminability_scores = self.calculate_feature_discriminability(features_scaled, labels)
        
        print("计算特征冗余度...")
        redundancy_scores = self.calculate_feature_redundancy(features_scaled)
        
        print("计算特征鲁棒性...")
        robustness_scores = self.calculate_feature_robustness(features_scaled, labels)
        
        # 综合评分
        # 权重可以根据需要调整
        weights = {
            'stability': 0.3,
            'discriminability': 0.3,
            'redundancy': 0.2,
            'robustness': 0.2
        }
        
        # 归一化评分
        stability_norm = (stability_scores - np.min(stability_scores)) / (np.max(stability_scores) - np.min(stability_scores) + 1e-8)
        discriminability_norm = (discriminability_scores - np.min(discriminability_scores)) / (np.max(discriminability_scores) - np.min(discriminability_scores) + 1e-8)
        redundancy_norm = (redundancy_scores - np.min(redundancy_scores)) / (np.max(redundancy_scores) - np.min(redundancy_scores) + 1e-8)
        robustness_norm = (robustness_scores - np.min(robustness_scores)) / (np.max(robustness_scores) - np.min(robustness_scores) + 1e-8)
        
        # 计算综合评分
        combined_scores = (weights['stability'] * stability_norm +
                          weights['discriminability'] * discriminability_norm +
                          weights['redundancy'] * redundancy_norm +
                          weights['robustness'] * robustness_norm)
        
        # 选择top-k特征
        top_indices = np.argsort(combined_scores)[::-1][:self.n_features]
        
        self.selected_features = top_indices
        self.feature_scores = {
            'combined': combined_scores,
            'stability': stability_scores,
            'discriminability': discriminability_scores,
            'redundancy': redundancy_scores,
            'robustness': robustness_scores
        }
        
        print(f"选择了 {len(top_indices)} 个最优特征")
        return top_indices
    
    def transform_features(self, features):
        """转换特征"""
        if self.selected_features is None:
            raise ValueError("请先调用 select_features 方法")
        
        return features[:, self.selected_features]
    
    def visualize_feature_selection(self, features=None, save_path=None):
        """可视化特征选择结果"""
        if self.feature_scores is None:
            print("没有特征评分数据")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('鲁棒特征选择分析', fontsize=16, fontweight='bold')
        
        # 1. 综合评分分布
        ax1 = axes[0, 0]
        combined_scores = self.feature_scores['combined']
        ax1.hist(combined_scores, bins=30, alpha=0.7, color='skyblue')
        ax1.axvline(np.sort(combined_scores)[-self.n_features], color='red', linestyle='--', 
                   label=f'Top {self.n_features} 阈值')
        ax1.set_xlabel('综合评分')
        ax1.set_ylabel('频次')
        ax1.set_title('综合评分分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 各维度评分对比
        ax2 = axes[0, 1]
        categories = ['稳定性', '区分度', '冗余度', '鲁棒性']
        scores = [np.mean(self.feature_scores['stability']),
                 np.mean(self.feature_scores['discriminability']),
                 np.mean(self.feature_scores['redundancy']),
                 np.mean(self.feature_scores['robustness'])]
        
        bars = ax2.bar(categories, scores, color=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax2.set_ylabel('平均评分')
        ax2.set_title('各维度平均评分')
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, score in zip(bars, scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 3. 特征评分散点图
        ax3 = axes[0, 2]
        stability = self.feature_scores['stability']
        discriminability = self.feature_scores['discriminability']
        
        # 区分选中和未选中的特征
        selected_mask = np.zeros(len(stability), dtype=bool)
        selected_mask[self.selected_features] = True
        
        ax3.scatter(stability[~selected_mask], discriminability[~selected_mask], 
                   alpha=0.5, color='lightgray', label='未选中')
        ax3.scatter(stability[selected_mask], discriminability[selected_mask], 
                   alpha=0.8, color='red', label='已选中')
        ax3.set_xlabel('稳定性评分')
        ax3.set_ylabel('区分度评分')
        ax3.set_title('稳定性 vs 区分度')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 特征相关性热力图（选中特征）
        ax4 = axes[1, 0]
        if len(self.selected_features) > 1 and features is not None:
            try:
                selected_features_data = self.transform_features(features)
                corr_matrix = np.corrcoef(selected_features_data.T)
                
                im = ax4.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
                ax4.set_title(f'选中特征相关性 ({len(self.selected_features)}个特征)')
                plt.colorbar(im, ax=ax4)
            except:
                ax4.text(0.5, 0.5, '无法计算相关性', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('特征相关性')
        else:
            ax4.text(0.5, 0.5, '特征数量不足或无特征数据', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('特征相关性')
        
        # 5. 评分排名
        ax5 = axes[1, 1]
        top_indices = self.selected_features
        top_scores = combined_scores[top_indices]
        
        ax5.bar(range(len(top_indices)), top_scores, color='lightcoral')
        ax5.set_xlabel('特征排名')
        ax5.set_ylabel('综合评分')
        ax5.set_title(f'Top {len(top_indices)} 特征评分')
        ax5.grid(True, alpha=0.3)
        
        # 6. 特征重要性分布
        ax6 = axes[1, 2]
        all_scores = combined_scores
        selected_scores = combined_scores[self.selected_features]
        
        ax6.hist(all_scores, bins=30, alpha=0.5, label='所有特征', color='lightblue')
        ax6.hist(selected_scores, bins=30, alpha=0.7, label='选中特征', color='red')
        ax6.set_xlabel('综合评分')
        ax6.set_ylabel('频次')
        ax6.set_title('特征重要性分布')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_feature_report(self):
        """生成特征选择报告"""
        if self.feature_scores is None:
            print("没有特征评分数据")
            return
        
        print("\n" + "="*60)
        print("鲁棒特征选择报告")
        print("="*60)
        
        print(f"\n📊 特征选择统计:")
        print(f"   总特征数: {len(self.feature_scores['combined'])}")
        print(f"   选中特征数: {len(self.selected_features)}")
        print(f"   选择比例: {len(self.selected_features)/len(self.feature_scores['combined'])*100:.1f}%")
        
        print(f"\n📈 评分统计:")
        combined_scores = self.feature_scores['combined']
        selected_scores = combined_scores[self.selected_features]
        
        print(f"   所有特征平均评分: {np.mean(combined_scores):.4f}")
        print(f"   选中特征平均评分: {np.mean(selected_scores):.4f}")
        print(f"   评分提升: {(np.mean(selected_scores)/np.mean(combined_scores)-1)*100:+.1f}%")
        
        print(f"\n🎯 各维度评分:")
        for metric, scores in self.feature_scores.items():
            if metric != 'combined':
                all_avg = np.mean(scores)
                selected_avg = np.mean(scores[self.selected_features])
                print(f"   {metric}: 所有={all_avg:.4f}, 选中={selected_avg:.4f}, "
                      f"提升={(selected_avg/all_avg-1)*100:+.1f}%")
        
        print(f"\n🔍 选中特征索引:")
        print(f"   {self.selected_features.tolist()}")
        
        print("\n" + "="*60)

def load_subject_features(subject_id, dataset='DB4', root_data='D:/DB4'):
    """加载受试者特征数据"""
    feature_file = os.path.join(root_data, f'data/restimulus/Fea/{dataset}_s{subject_id}feaEnhance.h5')
    
    if not os.path.exists(feature_file):
        print(f"特征文件不存在: {feature_file}")
        return None, None
    
    with h5py.File(feature_file, 'r') as f:
        features = f['features'][:]
        labels = f['label'][:]
    
    print(f"受试者 {subject_id} 特征数据: {features.shape}")
    return features, labels

def main():
    """主函数"""
    # 创建输出目录
    output_dir = Path("improvement")
    output_dir.mkdir(exist_ok=True)
    
    # 创建特征选择器
    selector = RobustFeatureSelector(n_features=50)
    
    # 加载多个受试者的数据进行特征选择
    subjects = [7, 8, 10]  # 包括低准确率和高准确率受试者
    all_features = []
    all_labels = []
    
    print("加载受试者特征数据...")
    for subject_id in subjects:
        features, labels = load_subject_features(subject_id)
        if features is not None:
            all_features.append(features)
            all_labels.append(labels)
            print(f"受试者 {subject_id}: {features.shape}")
    
    if not all_features:
        print("没有可用的特征数据")
        return
    
    # 合并所有受试者的数据
    combined_features = np.vstack(all_features)
    combined_labels = np.concatenate(all_labels)
    
    print(f"\n合并后数据形状: {combined_features.shape}")
    print(f"标签分布: {np.bincount(combined_labels)}")
    
    # 执行特征选择
    selected_indices = selector.select_features(combined_features, combined_labels)
    
    # 生成报告
    selector.generate_feature_report()
    
    # 可视化结果
    selector.visualize_feature_selection(save_path='improvement/robust_feature_selection.png')
    
    # 保存选中的特征索引
    np.save('improvement/selected_feature_indices.npy', selected_indices)
    print(f"\n选中特征索引已保存: improvement/selected_feature_indices.npy")
    
    print("\n鲁棒特征选择完成！")

if __name__ == "__main__":
    main() 