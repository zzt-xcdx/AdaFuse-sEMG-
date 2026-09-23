#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MobileOne模型重新参数化脚本
将训练时的多分支结构转换为推理时的单分支结构

使用方法:
python reparameterize_mobileone.py --model path/to/model.pth --output path/to/output.pth

注意:
- 训练完成后必须调用此脚本进行重新参数化
- 重新参数化后的模型参数量不变，但推理速度显著提升
- 重新参数化后的模型无法继续训练
"""

import argparse
import torch
import os
import sys

# 添加模型路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Models')))

from PyTorchModels_improved import MobileOneEnhancedModel

def reparameterize_model(model_path, output_path):
    """
    重新参数化MobileOne模型
    
    Args:
        model_path (str): 训练好的模型路径
        output_path (str): 重新参数化后的模型保存路径
    """
    print(f"正在加载模型: {model_path}")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 加载模型
    try:
        # 创建模型实例（推理模式）
        model = MobileOneEnhancedModel(
            emg_channels=16,  # 根据实际数据调整
            time_steps=1000,  # 根据实际数据调整
            feature_dim=64,   # 根据实际数据调整
            num_classes=12,   # 根据实际数据调整
            inference_mode=False  # 先设为False以加载权重
        )
        
        # 加载训练好的权重
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 如果保存的是state_dict，直接加载
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print("模型权重加载成功")
        
        # 打印重新参数化前的模型信息
        print("\n=== 重新参数化前 ===")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型总参数量: {total_params:,}")
        
        # 检查MobileOne分支的状态
        mobileone_branch = model.space_branch
        print(f"MobileOne分支推理模式: {mobileone_branch.inference_mode}")
        
        # 执行重新参数化
        print("\n开始重新参数化...")
        model.reparameterize()
        
        # 验证重新参数化结果
        print("\n=== 重新参数化后 ===")
        print(f"MobileOne分支推理模式: {mobileone_branch.inference_mode}")
        
        # 保存重新参数化后的模型
        print(f"\n保存重新参数化后的模型到: {output_path}")
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'emg_channels': 16,
                'time_steps': 1000,
                'feature_dim': 64,
                'num_classes': 12,
                'inference_mode': True
            },
            'reparameterized': True
        }, output_path)
        
        print("重新参数化完成！")
        print(f"模型已保存到: {output_path}")
        print("\n注意: 重新参数化后的模型无法继续训练，仅用于推理")
        
    except Exception as e:
        print(f"重新参数化过程中出现错误: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='MobileOne模型重新参数化')
    parser.add_argument('--model', type=str, required=True, help='训练好的模型路径')
    parser.add_argument('--output', type=str, required=True, help='重新参数化后的模型保存路径')
    
    args = parser.parse_args()
    
    try:
        reparameterize_model(args.model, args.output)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 