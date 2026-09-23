import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def to_bct(x: torch.Tensor, expected_c: int = None) -> torch.Tensor:
    """转换为 [B, C, T] 格式"""
    if x.dim() == 4 and x.size(-1) == 1:
        x = x[..., 0]
    if x.dim() != 3:
        raise ValueError(f"to_bct expects 3D or 4D with last=1, got shape {tuple(x.size())}")
    B, D1, D2 = x.shape
    if expected_c is not None:
        if D1 == expected_c:
            return x
        if D2 == expected_c:
            return x.permute(0, 2, 1)
    if D1 <= D2:
        return x
    else:
        return x.permute(0, 2, 1)


# ================================================================
# ① 借鉴 DSRANet：深度可分离卷积
# ================================================================
class DepthwiseSeparableConv1D(nn.Module):
    """1D 版本的深度可分离卷积（用于时域）"""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DepthwiseSeparableConv2D(nn.Module):
    """2D 版本的深度可分离卷积（用于空间）"""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# ================================================================
# ② 借鉴 DSRANet：多轴注意力（改造为 1D）
# ================================================================
class MultiAxisChannelAttention1D(nn.Module):
    """
    多轴通道注意力（1D 版本）
    用于时域分支，关注：
    - 电极维度（通道）
    - 时间维度
    """
    def __init__(self, channels, kernel_size=7, use_channel=True, use_time=True, dropout=0.0):
        super().__init__()
        self.use_channel = use_channel
        self.use_time = use_time
        padding = kernel_size // 2

        if self.use_time:
            self.conv_time = DepthwiseSeparableConv1D(
                channels, channels, kernel_size,
                padding=padding
            )
        if self.use_channel:
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Sequential(
                nn.Linear(channels, channels // 2),
                nn.SiLU(inplace=True),
                nn.Linear(channels // 2, channels)
            )
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # [B, C, T]
        B, C, T = x.size()
        attn = 1.0

        # 时间注意力
        if self.use_time:
            attn_t = self.sigmoid(self.conv_time(x))  # [B, C, T]
            attn = attn * attn_t

        # 通道注意力
        if self.use_channel:
            channel_avg = self.global_pool(x).view(B, C)  # [B, C]
            attn_c = self.fc(channel_avg)  # [B, C]
            attn_c = self.sigmoid(attn_c).view(B, C, 1)  # [B, C, 1]
            attn = attn * attn_c

        out = x * attn
        return self.dropout(out)


class MultiAxisChannelAttention2D(nn.Module):
    """
    多轴注意力（2D 版本）
    用于空间分支，关注：
    - 频率维度 (H)
    - 电极-时间维度 (W)
    - 通道维度 (C)
    """
    def __init__(self, channels, kernel_size=7, dropout=0.0):
        super().__init__()
        padding = kernel_size // 2

        # 频率维度注意力
        self.conv_freq = DepthwiseSeparableConv2D(
            channels, channels, kernel_size=(kernel_size, 1),
            padding=(padding, 0)
        )
        # 电极-时间维度注意力
        self.conv_time = DepthwiseSeparableConv2D(
            channels, channels, kernel_size=(1, kernel_size),
            padding=(0, padding)
        )
        # 通道注意力
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.SiLU(inplace=True),
            nn.Linear(channels // 2, channels)
        )
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # [B, C, H, W]
        B, C, H, W = x.size()

        # 频率 + 时间维度
        attn_spatial = self.sigmoid(self.conv_freq(x))  # [B, C, H, W]
        attn_spatial *= self.sigmoid(self.conv_time(x))

        # 通道维度
        channel_avg = self.global_pool(x).view(B, C)
        attn_channel = self.fc(channel_avg)
        attn_channel = self.sigmoid(attn_channel).view(B, C, 1, 1)

        out = x * attn_spatial * attn_channel
        return self.dropout(out)


# ================================================================
# ③ 借鉴 DSRANet：堆叠残差块
# ================================================================
class ResidualBlock1D(nn.Module):
    """1D 残差块"""
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dropout_rate=0.3):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv1D(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.silu = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

        self.conv2 = DepthwiseSeparableConv1D(
            out_channels, out_channels, kernel_size, 1, padding
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 跳接
        self.downsample = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.silu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        return self.silu(out)


class ResidualBlock2D(nn.Module):
    """2D 残差块"""
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dropout_rate=0.3):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv2D(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)

        self.conv2 = DepthwiseSeparableConv2D(
            out_channels, out_channels, kernel_size, 1, padding
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.silu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        return self.silu(out)


# ================================================================
# ④ 改进的 ECA1D
# ================================================================
class ECA1D(nn.Module):
    """高效通道注意力（1D）"""
    def __init__(self, channels, k_size=3, dropout=0.0):
        super().__init__()
        k = k_size if k_size % 2 == 1 else k_size + 1
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sig = nn.Sigmoid()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # [B, C, T]
        y = self.avg(x).transpose(1, 2)  # [B, 1, C]
        y = self.conv(y)  # [B, 1, C]
        y = self.sig(y).transpose(1, 2)  # [B, C, 1]
        y = self.drop(y)
        return x * y


# ================================================================
# ⑤ 改进的空间分支（借鉴 DSRANet 多阶段）
# ================================================================
class ImprovedSpatialBranchV2(nn.Module):
    """
    改进的空间分支：
    - 多阶段特征提取（4个阶段）
    - 每阶段包含：残差块 + 最大池化 + 多轴注意力
    - 输入：[B, 12, 600]
    - 输出：(B, 64)
    """
    def __init__(self, in_channels=12, dropout_rate=0.3):
        super().__init__()

        channels = [32, 64, 128, 256]
        dropout_rates = [0.5, 0.4, 0.3]

        # 初始卷积
        self.initial_conv = nn.Sequential(
            DepthwiseSeparableConv1D(in_channels, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm1d(channels[0]),
            nn.SiLU(inplace=True)
        )

        # 4 个阶段
        self.stages = nn.ModuleList()
        in_c = channels[0]
        for i in range(len(dropout_rates)):
            out_c = channels[i + 1]
            stage = nn.Sequential(
                ResidualBlock1D(in_c, out_c, dropout_rate=dropout_rates[i]),
                nn.MaxPool1d(kernel_size=2, stride=2),  # 时间降采样
                MultiAxisChannelAttention1D(out_c, kernel_size=7, dropout=dropout_rates[i])
            )
            self.stages.append(stage)
            in_c = out_c

        # 全局池化 + 映射
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(channels[-1], 128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64)
        )

    def forward(self, x):  # [B, C, T]
        x = self.initial_conv(x)  # [B, 32, T]
        for stage in self.stages:
            x = stage(x)  # 逐阶段降采样和特征提取
        x = self.global_pool(x).squeeze(-1)  # [B, 256]
        x = self.head(x)  # [B, 64]
        return x


# ================================================================
# ⑥ 改进的时域分支
# ================================================================
class ImprovedTemporalBranch(nn.Module):
    """
    改进的多尺度时域分支：
    - 多个并联分支（不同 kernel_size 和 dilation）
    - 每个分支：DepthwiseSeparableConv1D + 多轴注意力
    - 融合 + ECA1D 最终注意力
    - 输入：[B, 12, 600]
    - 输出：(B, 64)
    """
    def __init__(self, in_channels=12, out_dim=64,
                 kernel_sizes=(3, 5, 9), dilations=(1, 2, 3),
                 dropout=0.2):
        super().__init__()
        self.in_channels = in_channels
        num_branches = len(kernel_sizes)
        per_branch = max(16, out_dim // num_branches)

        branches = []
        for k, d in zip(kernel_sizes, dilations):
            pad = d * (k // 2)
            branch = nn.Sequential(
                # Depthwise: 沿时间卷积
                nn.Conv1d(in_channels, in_channels, kernel_size=k,
                         dilation=d, padding=pad, groups=in_channels, bias=False),
                nn.PReLU(),
                # Pointwise: 混合电极
                nn.Conv1d(in_channels, per_branch, kernel_size=1, bias=False),
                nn.PReLU(),
                nn.BatchNorm1d(per_branch),
                nn.Dropout(dropout)
            )
            branches.append(branch)
        self.branches = nn.ModuleList(branches)

        # 融合后的多轴注意力
        self.maca = MultiAxisChannelAttention1D(
            channels=per_branch * num_branches,
            kernel_size=7,
            use_time=True,
            use_channel=True,
            dropout=dropout
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(per_branch * num_branches, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):  # [B, C, T]
        x = to_bct(x, expected_c=self.in_channels)
        outs = [br(x) for br in self.branches]
        y = torch.cat(outs, dim=1)  # [B, per_branch*num_br, T]
        y = self.maca(y)  # 多轴注意力
        y = self.pool(y).squeeze(-1)  # [B, per_branch*num_br]
        y = self.head(y)  # [B, out_dim]
        return y


# ================================================================
# ⑦ 改进的时频分支（保持原样，但加强注意力）
# ================================================================
class ImprovedLinearResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.PReLU()
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        identity = x
        out = self.linear(x)
        out = self.relu(out)
        out = self.bn(out)
        out = self.dropout(out)
        out += self.shortcut(identity)
        return out


class ImprovedTimeFreqBranch(nn.Module):
    """
    改进的时频分支：
    - 多层线性残差块
    - 输入：[B, feat_dim]
    - 输出：(B, 64)
    """
    def __init__(self, feature_dim, out_dim=64, dropout_rate=0.25):
        super().__init__()
        self.backbone = nn.Sequential(
            ImprovedLinearResidualBlock(feature_dim, 128, dropout_rate),
            ImprovedLinearResidualBlock(128, 256, dropout_rate),
            ImprovedLinearResidualBlock(256, 128, dropout_rate),
            ImprovedLinearResidualBlock(128, out_dim, dropout_rate)
        )

    def forward(self, x):
        return self.backbone(x)


# ================================================================
# ⑧ 最终融合模型（改进版）
# ================================================================
class ImprovedFeaAndEmg_V2(nn.Module):
    """
    改进的三分支模型（融合 DSRANet 思想）：
    - 分支1：ImprovedSpatialBranchV2 → 64
    - 分支2：ImprovedTemporalBranch → 64
    - 分支3：ImprovedTimeFreqBranch → 64
    - 门控融合：softmax 权重
    - 分类器：(B, 192) → num_classes
    """
    def __init__(self, emg_channels, time_steps, feature_dim,
                 num_classes, dropout_rate=0.25):
        super().__init__()

        # 三个分支
        self.space_branch = ImprovedSpatialBranchV2(
            in_channels=emg_channels,
            dropout_rate=dropout_rate
        )
        self.time_domain_branch = ImprovedTemporalBranch(
            in_channels=emg_channels,
            out_dim=64,
            kernel_sizes=(3, 5, 9),
            dilations=(1, 2, 3),
            dropout=dropout_rate * 0.8
        )
        self.time_freq_branch = ImprovedTimeFreqBranch(
            feature_dim=feature_dim,
            out_dim=64,
            dropout_rate=dropout_rate
        )

        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(64 * 3, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(64 * 3, 256),
            nn.PReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

        self.emg_channels = emg_channels
        self.ablate_branch_default = None

    def set_ablation(self, branch: str | None):
        """设置消融分支"""
        if not branch:
            self.ablate_branch_default = None
        else:
            n = str(branch).strip().lower()
            if 'space' in n:
                self.ablate_branch_default = 'space'
            elif 'time_dom' in n or 'temporal' in n:
                self.ablate_branch_default = 'time_dom'
            elif 'time_freq' in n or 'freq' in n:
                self.ablate_branch_default = 'time_freq'
            else:
                self.ablate_branch_default = None
        print(f"[Model] 消融设置: {self.ablate_branch_default or 'none'}")

    def forward(self, feature_data, emg_data,
                return_gate_weights: bool = False,
                ablate_branch: str | None = None):
        """前向传播"""
        # 统一 EMG 格式
        emg_bct = to_bct(emg_data, expected_c=self.emg_channels)  # [B, C, T]

        # 三个分支特征
        space_f = self.space_branch(emg_bct)          # (B, 64)
        time_dom_f = self.time_domain_branch(emg_bct)  # (B, 64)
        time_freq_f = self.time_freq_branch(feature_data)  # (B, 64)

        # 消融处理
        ab = ablate_branch
        if ab is None:
            ab = self.ablate_branch_default
        if ab == 'space':
            space_f = torch.zeros_like(space_f)
        elif ab == 'time_dom':
            time_dom_f = torch.zeros_like(time_dom_f)
        elif ab == 'time_freq':
            time_freq_f = torch.zeros_like(time_freq_f)

        # 门控
        gate_in = torch.cat([space_f, time_dom_f, time_freq_f], dim=1)  # (B, 192)
        gate_weights = self.gate(gate_in)  # (B, 3)

        # 消融后重新归一化
        if ab is not None:
            idx = {'space': 0, 'time_dom': 1, 'time_freq': 2}[ab]
            gate_weights = gate_weights.clone()
            gate_weights[:, idx] = 0.0
            gate_weights = gate_weights / (gate_weights.sum(dim=1, keepdim=True) + 1e-8)

        # 加权融合
        s = space_f * gate_weights[:, 0:1]
        td = time_dom_f * gate_weights[:, 1:2]
        tf = time_freq_f * gate_weights[:, 2:3]
        final_f = torch.cat([s, td, tf], dim=1)  # (B, 192)

        # 分类
        output = self.classifier(final_f)

        if return_gate_weights:
            return output, gate_weights
        else:
            return output


if __name__ == '__main__':
    # 测试
    batch_size = 4
    emg_channels = 12
    time_steps = 600
    feature_dim = 64
    num_classes = 53

    model = ImprovedFeaAndEmg_V2(
        emg_channels=emg_channels,
        time_steps=time_steps,
        feature_dim=feature_dim,
        num_classes=num_classes,
        dropout_rate=0.25
    )

    # 测试输入
    emg_data = torch.randn(batch_size, emg_channels, time_steps, 1)
    feat_data = torch.randn(batch_size, feature_dim)

    output, gate_w = model(feat_data, emg_data, return_gate_weights=True)
    print(f"输出形状: {output.shape}")
    print(f"门控权重: {gate_w.shape}")
    print(f"门控均值: Space={gate_w[:, 0].mean():.3f}, "
          f"TimeDom={gate_w[:, 1].mean():.3f}, "
          f"TimeFreq={gate_w[:, 2].mean():.3f}")

    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")