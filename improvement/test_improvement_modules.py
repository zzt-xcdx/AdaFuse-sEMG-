#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试改进模块
验证自适应信号增强和鲁棒特征选择模块是否正常工作
"""

import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_signal_enhancer():
    """测试信号增强器"""
    print("🔧 测试自适应信号增强器...")
    
    try:
        from improvement.adaptive_signal_enhancement import AdaptiveSignalEnhancer
        
        # 创建增强器
        enhancer = AdaptiveSignalEnhancer(fs=2000)
        
        # 生成测试信号
        t = np.linspace(0, 1, 2000)
        test_signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.random.normal(0, 1, 2000)
        test_emg = np.column_stack([test_signal, test_signal * 0.8, test_signal * 1.2])
        
        # 测试质量评估
        quality_metrics = enhancer.assess_signal_quality(test_emg)
        print(f"✅ 信号质量评估正常: {len(quality_metrics)} 个通道")
        
        # 测试自适应滤波
        enhanced_emg = enhancer.adaptive_filter(test_emg, quality_metrics)
        print(f"✅ 自适应滤波正常: 输出形状 {enhanced_emg.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 信号增强器测试失败: {e}")
        return False

def test_feature_selector():
    """测试特征选择器"""
    print("\n🎯 测试鲁棒特征选择器...")
    
    try:
        from improvement.robust_feature_selection import RobustFeatureSelector
        
        # 创建特征选择器
        selector = RobustFeatureSelector(n_features=10)
        
        # 生成测试特征和标签
        n_samples = 1000
        n_features = 50
        test_features = np.random.normal(0, 1, (n_samples, n_features))
        test_labels = np.random.randint(0, 5, n_samples)
        
        # 测试特征选择
        selected_indices = selector.select_features(test_features, test_labels)
        print(f"✅ 特征选择正常: 选择了 {len(selected_indices)} 个特征")
        
        # 测试特征转换
        transformed_features = selector.transform_features(test_features)
        print(f"✅ 特征转换正常: 输出形状 {transformed_features.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 特征选择器测试失败: {e}")
        return False

def test_data_loading():
    """测试数据加载"""
    print("\n📁 测试数据加载...")
    
    try:
        from improvement.robust_feature_selection import load_subject_features
        
        # 检查DB4数据是否存在
        test_subject = 7
        feature_file = f'D:/DB4/data/restimulus/Fea/DB4_s{test_subject}feaEnhance.h5'
        
        if os.path.exists(feature_file):
            features, labels = load_subject_features(test_subject, 'DB4', 'D:/DB4')
            if features is not None:
                print(f"✅ 数据加载正常: 受试者{test_subject} 特征形状 {features.shape}")
                return True
            else:
                print(f"❌ 数据加载失败: 无法读取特征文件")
                return False
        else:
            print(f"❌ 数据文件不存在: {feature_file}")
            return False
            
    except Exception as e:
        print(f"❌ 数据加载测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 开始测试改进模块...")
    print("="*50)
    
    # 创建输出目录
    output_dir = Path("improvement")
    output_dir.mkdir(exist_ok=True)
    
    # 运行测试
    tests = [
        ("信号增强器", test_signal_enhancer),
        ("特征选择器", test_feature_selector),
        ("数据加载", test_data_loading)
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    # 输出测试结果
    print("\n" + "="*50)
    print("📊 测试结果总结:")
    print("="*50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 所有测试通过！改进模块可以正常使用。")
        print("\n🚀 建议下一步:")
        print("  1. 运行 python improvement/complete_improvement_pipeline.py")
        print("  2. 或者分别运行各个模块进行改进")
    else:
        print("⚠️  部分测试失败，请检查错误信息并修复问题。")
    
    return all_passed

if __name__ == "__main__":
    main() 