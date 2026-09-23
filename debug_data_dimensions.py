import os
import h5py
import numpy as np
import torch
import sys

# 添加模型路径
sys.path.append('Models')

from PyTorchModels_improved import ImprovedFeaAndEmg_se_PyTorch

def debug_data_dimensions():
    """调试数据维度和模型参数"""
    
    # 模拟数据路径
    root_data = "data"
    dataset = "DB4"
    subject_id = 1
    
    try:
        # 尝试加载数据
        emg_file = os.path.join(root_data, f'data/restimulus/reSegEnhance/{dataset}_s{subject_id}SegEnhance.h5')
        feature_file = os.path.join(root_data, f'data/restimulus/Fea/{dataset}_s{subject_id}feaEnhance.h5')
        
        print(f"检查文件是否存在:")
        print(f"EMG文件: {emg_file} - {'存在' if os.path.exists(emg_file) else '不存在'}")
        print(f"特征文件: {feature_file} - {'存在' if os.path.exists(feature_file) else '不存在'}")
        
        if os.path.exists(emg_file) and os.path.exists(feature_file):
            print("\n加载数据...")
            with h5py.File(emg_file, 'r') as f:
                emg_data = f['emg'][:]
                labels = f['label'][:]
                rep_arr = f['rep'][:]
            with h5py.File(feature_file, 'r') as f:
                feature_data = f['features'][:]
            
            print(f"原始数据维度:")
            print(f"EMG数据: {emg_data.shape}")
            print(f"特征数据: {feature_data.shape}")
            print(f"标签数据: {labels.shape}")
            print(f"重复数组: {rep_arr.shape}")
            
            # 模拟数据分割
            rep_vali = 2
            # 简单分割：前80%训练，10%验证，10%测试
            n_samples = len(emg_data)
            n_train = int(0.8 * n_samples)
            n_val = int(0.1 * n_samples)
            
            emg_train = emg_data[:n_train]
            emg_vali = emg_data[n_train:n_train+n_val]
            emg_test = emg_data[n_train+n_val:]
            
            feature_train = feature_data[:n_train]
            feature_vali = feature_data[n_train:n_train+n_val]
            feature_test = feature_data[n_train+n_val:]
            
            label_train = labels[:n_train]
            label_vali = labels[n_train:n_train+n_val]
            label_test = labels[n_train+n_val:]
            
            print(f"\n分割后数据维度:")
            print(f"训练集 - EMG: {emg_train.shape}, 特征: {feature_train.shape}, 标签: {label_train.shape}")
            print(f"验证集 - EMG: {emg_vali.shape}, 特征: {feature_vali.shape}, 标签: {label_vali.shape}")
            print(f"测试集 - EMG: {emg_test.shape}, 特征: {feature_test.shape}, 标签: {label_test.shape}")
            
            print(f"\n模型参数:")
            print(f"emg_channels (emg_train.shape[2]): {emg_train.shape[2]}")
            print(f"time_steps (emg_train.shape[1]): {emg_train.shape[1]}")
            print(f"feature_dim (feature_train.shape[1]): {feature_train.shape[1]}")
            print(f"num_classes: {np.max(label_train)+1}")
            
            # 创建模型
            dropout_rate = 0.4
            model = ImprovedFeaAndEmg_se_PyTorch(
                emg_channels=emg_train.shape[2],
                time_steps=emg_train.shape[1], 
                feature_dim=feature_train.shape[1],
                num_classes=np.max(label_train)+1,
                dropout_rate=dropout_rate
            )
            
            print(f"\n模型结构:")
            print(f"空间分支参数: {sum(p.numel() for p in model.space_branch.parameters())}")
            print(f"时频分支参数: {sum(p.numel() for p in model.time_freq_branch.parameters())}")
            print(f"门控参数: {sum(p.numel() for p in model.gate.parameters())}")
            print(f"分类器参数: {sum(p.numel() for p in model.classifier.parameters())}")
            
            # 检查时频分支的第一个线性层
            first_block = model.time_freq_branch[0]
            print(f"\n第一个残差块:")
            print(f"线性层输入维度: {first_block.linear.in_features}")
            print(f"线性层输出维度: {first_block.linear.out_features}")
            
            # 测试前向传播
            print(f"\n测试前向传播...")
            batch_size = 2
            test_feature = torch.randn(batch_size, feature_train.shape[1])
            test_emg = torch.randn(batch_size, emg_train.shape[1], emg_train.shape[2], 1)
            
            print(f"测试输入维度:")
            print(f"特征输入: {test_feature.shape}")
            print(f"EMG输入: {test_emg.shape}")
            
            try:
                with torch.no_grad():
                    output, gate_weights = model(test_feature, test_emg, return_gate_weights=True)
                print(f"前向传播成功!")
                print(f"输出维度: {output.shape}")
                print(f"门控权重维度: {gate_weights.shape}")
            except Exception as e:
                print(f"前向传播失败: {e}")
                import traceback
                traceback.print_exc()
                
        else:
            print("数据文件不存在，无法进行测试")
            
    except Exception as e:
        print(f"调试过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_data_dimensions() 