import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# 改进的空间特征提取器，使用Conv→ReLU→BN顺序
class ImprovedSpatialFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels=64, dropout_rate=0.3):
        super(ImprovedSpatialFeatureExtractor, self).__init__()
        self.spatial_net = nn.Sequential(
            # 第一层：使用改进的顺序
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=1, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(16),
            
            
            # 第二层：使用改进的顺序
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(32),  # 激活函数在BN之前
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate),
            
            # 第三层：使用改进的顺序
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),  # 激活函数在BN之前
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, out_channels))
        self.projection = nn.Linear(64, out_channels)
    
    def forward(self, x):
        B, C, T, D = x.shape
        x = x.permute(0, 3, 1, 2)  # (B, 1, C, T)
        x = self.spatial_net(x)
        x = self.adaptive_pool(x)
        x = x.view(B, 64, -1)
        x = x.mean(dim=2)
        x = self.projection(x)
        return x

# 改进的线性残差块，使用Linear→ReLU→BN顺序
class ImprovedLinearResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate):
        super(ImprovedLinearResidualBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.PReLU()  # 激活函数在BN之前
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
    
    def forward(self, x):
        # 确保输入是2D张量 (batch_size, features)
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        # 确保所有组件都在同一设备上
        device = x.device
        self.linear = self.linear.to(device)
        self.bn = self.bn.to(device)
        self.shortcut = self.shortcut.to(device)
        
        identity = x
        out = self.linear(x)
        out = self.relu(out)  # 激活函数在BN之前
        out = self.bn(out)
        out = self.dropout(out)
        out += self.shortcut(identity)
        return out

# 改进的EMA模块
class ImprovedEMA(nn.Module):
    def __init__(self, channels, N=1, c2=None, factor=16, dropout_rate=0.3):
        super(ImprovedEMA, self).__init__()
        # 确保factor不超过通道数
        self.groups = min(factor, channels)
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.dropout = nn.Dropout(dropout_rate)
        
        # 使用改进的空间特征提取器
        self.spatial_extractor = ImprovedSpatialFeatureExtractor(
            in_channels=channels, 
            out_channels=64,
            dropout_rate=dropout_rate
        )

        # 坐标注意力模块
        self.pool_c = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_t = nn.AdaptiveAvgPool2d((1, None))
        self.conv1x1 = nn.Conv2d(N, N, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.agp = nn.AdaptiveAvgPool2d((1, None))
        
        # 只保留一个 PReLU
        self.prelu = nn.PReLU()

    def forward(self, x):
        # 记住输入张量的设备
        device = x.device
        
        # 检查输入的维度，并进行必要的调整
        if len(x.size()) == 2:  # 如果输入是二维的 (B, F)
            return x  # 直接返回，不做处理
        
        # 如果是4维输入 (B, C, T, 1)，使用改进的空间特征提取
        if len(x.size()) == 4:
            B, C, T, D = x.size()
            # 确保最后一个维度是1
            if D != 1:
                raise ValueError(f"Expected last dimension to be 1, got {D}")
            
            # 使用改进的空间特征提取器提取特征
            return self.spatial_extractor(x)
            
        # 原始的EMA处理逻辑，用于处理标准形状的输入
        B, N, T, C = x.size()  # 原始代码期望的4维输入
        channel_group = C // self.groups

        # 坐标注意力模块
        group_x = x.reshape(B * self.groups, N, T, -1)
        x_t = self.pool_t(group_x)
        x_c = self.pool_c(group_x).permute(0, 1, 3, 2)
        
        # 特征融合
        tc = self.conv1x1(torch.cat([x_t, x_c], dim=-1))
        tc = self.prelu(tc)
        tc = self.dropout(tc)
        
        x_t, x_c = torch.split(tc, [channel_group, T], dim=-1)

        # 1×1分支和3×3分支
        x1 = group_x * x_t.sigmoid() * x_c.permute(0, 1, 3, 2).sigmoid()
        x2 = self.conv3x3(group_x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # 跨空间学习
        x11 = self.softmax(self.agp(x1))
        x12 = x2.permute(0, 1, 3, 2)
        y1 = torch.matmul(x11, x12)

        x21 = self.softmax(self.agp(x2))
        x22 = x1.permute(0, 1, 3, 2)
        y2 = torch.matmul(x21, x22)

        # 聚合空间信息
        weights = (y1 + y2).reshape(B * self.groups, N, T, 1)
        weights_ = weights.sigmoid()
        out = (group_x * weights_).reshape(B, N, T, C)
        
        return out

# 使用备份文件中的门控融合模块
class GatedFusion(nn.Module):
    def __init__(self, dim):
        super(GatedFusion, self).__init__()
        self.fc1 = nn.Linear(dim, dim, bias=True)
        self.fc2 = nn.Linear(dim, dim, bias=True)

    def forward(self, x1, x2):
        # 确保输入的形状一致，因为门控融合是在展平的特征上进行的
        if x1.shape != x2.shape:
             raise ValueError(f"Input shapes mismatch for GatedFusion: {x1.shape} vs {x2.shape}")

        x11 = self.fc1(x1)
        x22 = self.fc2(x2)
        # 通过门控单元生成权重表示
        z = torch.sigmoid(x11 + x22)
        # 对两部分输入执行加权和
        out = z * x1 + (1 - z) * x2
        return out

# 改进的PyTorch版本FeaAndEmg_se模型，使用备份文件中的门控机制
class ImprovedFeaAndEmg_se_PyTorch(nn.Module):
    def __init__(self, emg_channels, time_steps, feature_dim, num_classes, dropout_rate=0.25):
        super().__init__()
        # 空间分支用改进的EMA，输出(B, 64)
        self.space_branch = ImprovedEMA(emg_channels, time_steps)
        
        # 时频分支，输出(B, 64) - 使用备份文件的结构
        self.time_freq_branch = nn.Sequential(
            ImprovedLinearResidualBlock(feature_dim, 64, dropout_rate),
            ImprovedLinearResidualBlock(64, 128, dropout_rate),
            ImprovedLinearResidualBlock(128, 64, dropout_rate)
        )
        
        # 门控融合 - 使用备份文件中的门控机制
        self.gate = nn.Sequential(
            nn.Linear(128, 2),  # 64+64=128
            nn.Softmax(dim=-1)
        )
        
        # 分类器 - 使用备份文件中的分类器结构
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.PReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, feature_data, emg_data, return_gate_weights=False):
        # 空间特征分支 (处理原始 EMG 信号) - 输出64维
        space_features = self.space_branch(emg_data)
        
        # 时频特征处理分支 - 输出64维
        time_freq_features = self.time_freq_branch(feature_data)
        
        # 融合特征
        combined_features = torch.cat([space_features, time_freq_features], dim=1)
        
        # 门控权重
        gate_weights = self.gate(combined_features)
        
        # 加权融合
        gated_space_features = space_features * gate_weights[:, 0].unsqueeze(1)
        gated_time_freq_features = time_freq_features * gate_weights[:, 1].unsqueeze(1)
        
        final_features = torch.cat([gated_space_features, gated_time_freq_features], dim=1)
            
        # 分类
        output = self.classifier(final_features)
        
        if return_gate_weights:
            return output, gate_weights
        else:
            return output

# 其他分类器保持不变...