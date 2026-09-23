import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def to_bct(x: torch.Tensor, expected_c: int = None) -> torch.Tensor:
    """
    将输入 EMG 张量转换为 [B, C, T] 形状。
    支持输入形状：[B, T, C]、[B, C, T]、[B, C, T, 1]
    """
    if x.dim() == 4 and x.size(-1) == 1:
        x = x[..., 0]  # [B, C, T, 1] -> [B, C, T]
    if x.dim() != 3:
        raise ValueError(f"to_bct expects 3D or 4D with last=1, got shape {tuple(x.size())}")

    B, D1, D2 = x.shape  # 可能是 [B, C, T] 或 [B, T, C]
    if expected_c is not None:
        if D1 == expected_c:
            return x  # 已经是 [B, C, T]
        if D2 == expected_c:
            return x.permute(0, 2, 1)  # [B, T, C] -> [B, C, T]

    if D1 <= D2:
        return x  # 认为 [B, C, T]
    else:
        return x.permute(0, 2, 1)  # 认为 [B, T, C] -> [B, C, T]


class ECA1D(nn.Module):
    """
    高效通道注意力（ECA），用于 1D 序列的"通道注意力"（对电极通道加权）。
    输入:  x: [B, C, T]
    输出:  y: [B, C, T] （加权后的结果）
    """
    def __init__(self, channels, k_size=3, dropout=0.0):
        super().__init__()
        k = k_size if k_size % 2 == 1 else k_size + 1
        self.avg = nn.AdaptiveAvgPool1d(1)          # 沿 T 聚合 -> [B, C, 1]
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sig = nn.Sigmoid()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, C, T]
        y = self.avg(x).transpose(1, 2)  # [B, 1, C]
        y = self.conv(y)                 # [B, 1, C]
        y = self.sig(y).transpose(1, 2)  # [B, C, 1]
        y = self.drop(y)
        return x * y                     # 广播到 [B, C, T]


class ImprovedSpatialFeatureExtractor(nn.Module):
    """
    CEINet
    输入:  (B, C, T, 1) 先转为 (B, 1, C, T)
    输出:  (B, out_channels)
    """
    def __init__(self, in_channels, out_channels=64, dropout_rate=0.3):
        super(ImprovedSpatialFeatureExtractor, self).__init__()
        self.spatial_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=1, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate),

            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, out_channels))
        self.projection = nn.Linear(64, out_channels)

    def forward(self, x):
        B, C, T, D = x.shape
        x = x.permute(0, 3, 1, 2)  # (B, 1, C, T)
        x = self.spatial_net(x)    # (B, 64, C/4, T/4)
        x = self.adaptive_pool(x)  # (B, 64, 1, out_channels)
        x = x.view(B, 64, -1)      # (B, 64, out_channels)
        x = x.mean(dim=2)          # (B, 64)
        x = self.projection(x)     # (B, out_channels)
        return x


class ImprovedLinearResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate):
        super(ImprovedLinearResidualBlock, self).__init__()
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


class ImprovedEMA(nn.Module):
    """
    空间分支（CEINet）：
    - 接受 (B, C, T, 1) 或 (B, C, T) 或 (B, T, C)
    - 统一转换为 [B, C, T]，先做 ECA1D 通道注意力，再转为 4D 进入空间 CNN -> (B, 64)
    """
    def __init__(self, channels, N=1, c2=None, factor=16, dropout_rate=0.3):
        super(ImprovedEMA, self).__init__()
        self.in_channels = channels
        self.groups = min(factor, channels)
        assert channels // self.groups > 0
        self.dropout = nn.Dropout(dropout_rate)

        # 通道注意力（对电极通道加权）
        self.eca = ECA1D(channels, k_size=3, dropout=dropout_rate)

        # 空间特征提取器
        self.spatial_extractor = ImprovedSpatialFeatureExtractor(
            in_channels=channels,
            out_channels=64,
            dropout_rate=dropout_rate
        )

        # 以下为旧的占位，保兼容（不使用）
        self.pool_c = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_t = nn.AdaptiveAvgPool2d((1, None))
        self.conv1x1 = nn.Conv2d(N, N, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.agp = nn.AdaptiveAvgPool2d((1, None))
        self.prelu = nn.PReLU()

    def forward(self, x):
        # 统一转换为 [B, C, T]
        if x.dim() == 4 and x.size(-1) == 1:
            x = x[..., 0]  # [B, C, T, 1] -> [B, C, T]
        x = to_bct(x, expected_c=self.in_channels)  # [B, C, T]

        # 通道注意力 -> 变 4D -> 空间 CNN
        xt = self.eca(x)               # [B, C, T]
        xt4 = xt.unsqueeze(-1)         # [B, C, T, 1]
        return self.spatial_extractor(xt4)  # (B, 64)


class MultiScaleTemporal1D(nn.Module):
    """
    DTANet（可配置版本）
    支持三种对照以隔离 MS-DSC 的贡献：
    1) 'ms_dsc'     : 多尺度 depthwise separable（你当前的结构）
    2) 'single_dsc': 单尺度 depthwise separable（隔离"多尺度"带来的增益）
    3) 'standard'  : 标准 Conv1d（替代 depthwise separable）

    输入:  任意 [B, T, C]/[B, C, T]/[B, C, T, 1]
    输出:  (B, out_dim) 默认 64
    """
    def __init__(
        self,
        in_channels,
        out_dim=64,
        kernel_sizes=(3, 5, 9),
        dilations=(1, 2, 3),
        dropout=0.2,
        variant: str = "ms_dsc",   # 'ms_dsc' | 'single_dsc' | 'standard'
    ):
        super().__init__()
        self.in_channels = in_channels
        self.variant = str(variant).lower().strip()

        if self.variant not in ("ms_dsc", "single_dsc", "standard"):
            raise ValueError(f"Unknown dtanet variant: {variant}")

        if self.variant == "ms_dsc":
            assert len(kernel_sizes) == len(dilations)
            num_br = len(kernel_sizes)
            ks_ds = list(kernel_sizes)
            ds_ds = list(dilations)
        else:
            # 单尺度/标准卷积：只保留一个尺度（取第一个配置）
            num_br = 1
            ks_ds = [kernel_sizes[0]]
            ds_ds = [dilations[0]]

        per_branch = max(16, out_dim // num_br)

        branches = []
        for k, d in zip(ks_ds, ds_ds):
            pad = d * (k // 2)

            if self.variant in ("ms_dsc", "single_dsc"):
                # depthwise: 沿时间卷积，不混合电极通道
                depthwise = nn.Conv1d(
                    in_channels, in_channels,
                    kernel_size=k, dilation=d,
                    padding=pad, groups=in_channels, bias=False
                )
                # pointwise: 混合电极通道/投影到 per_branch
                pointwise = nn.Conv1d(
                    in_channels, per_branch,
                    kernel_size=1, bias=False
                )
                branch = nn.Sequential(
                    depthwise,
                    nn.PReLU(),
                    pointwise,
                    nn.PReLU(),
                    nn.BatchNorm1d(per_branch),
                    nn.Dropout(dropout),
                )
            else:
                # standard conv：直接用普通 Conv1d（不做 depthwise separable）
                branch = nn.Sequential(
                    nn.Conv1d(
                        in_channels, per_branch,
                        kernel_size=k, dilation=d,
                        padding=pad, bias=False
                    ),
                    nn.PReLU(),
                    nn.BatchNorm1d(per_branch),
                    nn.Dropout(dropout),
                )

            branches.append(branch)

        self.branches = nn.ModuleList(branches)

        # 对融合后的通道做 ECA
        fused_channels = per_branch * num_br
        self.eca_after = ECA1D(channels=fused_channels, k_size=3, dropout=dropout)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(fused_channels, out_dim)

    def forward(self, x):
        # 统一 [B, C, T]
        if x.dim() == 4 and x.size(-1) == 1:
            x = x[..., 0]
        x = to_bct(x, expected_c=self.in_channels)  # [B, C, T]

        outs = []
        for br in self.branches:
            outs.append(br(x))  # (B, per_branch, T)

        y = torch.cat(outs, dim=1)          # (B, fused_channels, T)
        y = self.eca_after(y)              # (B, fused_channels, T)
        y = self.pool(y).squeeze(-1)       # (B, fused_channels)
        y = self.head(y)                   # (B, out_dim)
        return y


class GatedFusion(nn.Module):
    def __init__(self, dim):
        super(GatedFusion, self).__init__()
        self.fc1 = nn.Linear(dim, dim, bias=True)
        self.fc2 = nn.Linear(dim, dim, bias=True)

    def forward(self, x1, x2):
        if x1.shape != x2.shape:
            raise ValueError(f"Input shapes mismatch for GatedFusion: {x1.shape} vs {x2.shape}")
        z = torch.sigmoid(self.fc1(x1) + self.fc2(x2))
        return z * x1 + (1 - z) * x2


class ImprovedFeaAndEmg_se_PyTorch(nn.Module):
    """
    最终模型（三分支 + 门控融合）：
    - 分支1：CEINet(ImprovedEMA) -> 64
    - 分支2：时频分支（线性残差 MLP）-> 64
    - 分支3：DTANet多尺度时域分支 -> 64

    门控：对 concat([64,64,64])=192 生成 3 个 softmax 权重
    融合：按权重缩放后再 concat -> 192
    分类器输入维度为 192

    forward(feature_data, emg_data[, return_gate_weights, ablate_branch])
    """
    def __init__(
        self,
        emg_channels,
        time_steps,
        feature_dim,
        num_classes,
        dropout_rate=0.25,
        dtanet_variant: str = "ms_dsc",  # 'ms_dsc' | 'single_dsc' | 'standard'
    ):
        super().__init__()

        # 分支1：空间分支
        self.space_branch = ImprovedEMA(emg_channels, time_steps, dropout_rate=dropout_rate)

        # 分支3：多尺度时域分支（可变对照）
        self.time_domain_branch = MultiScaleTemporal1D(
            in_channels=emg_channels,
            out_dim=64,
            kernel_sizes=(3, 5, 9),
            dilations=(1, 2, 3),
            dropout=dropout_rate * 0.8,
            variant=dtanet_variant,
        )

        # 分支2：时频分支
        self.time_freq_branch = nn.Sequential(
            ImprovedLinearResidualBlock(feature_dim, 64, dropout_rate),
            ImprovedLinearResidualBlock(64, 128, dropout_rate),
            ImprovedLinearResidualBlock(128, 64, dropout_rate),
        )

        # 三分支门控（用 concat 后的 192 作为门控输入）
        self.gate = nn.Sequential(
            nn.Linear(64 * 3, 3),
            nn.Softmax(dim=-1),
        )

        # 分类器：输入 192
        self.classifier = nn.Sequential(
            nn.Linear(64 * 3, 128),
            nn.PReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
        )

        self.emg_channels = emg_channels  # 用于形状校正

        # —模型实例的默认消融开关（不传参数时使用）
        self.ablate_branch_default = None  # 可选: None/'space'/'time_dom'/'time_freq'

    @staticmethod
    def _normalize_branch_name(name: str | None):
        """把多种写法统一到: None | 'space' | 'time_dom' | 'time_freq'"""
        if not name:
            return None
        n = str(name).strip().lower()
        if n in ('space',):
            return 'space'
        if n in ('time_dom', 'time-domain', 'time_domain', 'temporal', 'time'):
            return 'time_dom'
        if n in ('time_freq', 'time-frequency', 'time_frequency', 'tf', 'freq'):
            return 'time_freq'
        return None

    def set_ablation(self, branch: str | None):
        """
        设置默认的消融分支（写一次，后续 forward 不传也生效）
        可选: None / 'space' / 'time_dom' / 'time_freq'
        """
        self.ablate_branch_default = self._normalize_branch_name(branch)
        print(f"[Model] 消融设置: {self.ablate_branch_default or 'none'}")

    def forward(
        self,
        feature_data,
        emg_data,
        return_gate_weights: bool = False,
        ablate_branch: str | None = None
    ):
        """
        ablate_branch: 若传入则覆盖模型默认设置；不传则使用 self.ablate_branch_default
        """
        # 统一 EMG 到 [B, C, T]，避免分支里重复判定
        emg_bct = to_bct(emg_data, expected_c=self.emg_channels)  # [B, C, T]

        # 分支特征
        space_features = self.space_branch(emg_bct)               # (B, 64)
        time_dom_features = self.time_domain_branch(emg_bct)      # (B, 64)
        time_freq_features = self.time_freq_branch(feature_data)  # (B, 64)

        # —— 消融：先把指定分支特征置零，避免门控看到其信息
        ab = self._normalize_branch_name(ablate_branch)  # 以调用参数优先
        if ab is None:
            ab = self.ablate_branch_default             # 否则用模型默认设置

        if ab == 'space':
            space_features = torch.zeros_like(space_features)
        elif ab == 'time_dom':
            time_dom_features = torch.zeros_like(time_dom_features)
        elif ab == 'time_freq':
            time_freq_features = torch.zeros_like(time_freq_features)

        # 门控输入
        gate_in = torch.cat([space_features, time_dom_features, time_freq_features], dim=1)  # (B, 192)
        gate_weights = self.gate(gate_in)  # (B, 3) softmax 后

        # —— 若做了消融：把该分支权重置 0，并对剩余两支重归一化（总和=1）
        if ab is not None:
            idx = {'space': 0, 'time_dom': 1, 'time_freq': 2}[ab]
            gate_weights = gate_weights.clone()
            gate_weights[:, idx] = 0.0
            gate_weights = gate_weights / (gate_weights.sum(dim=1, keepdim=True) + 1e-8)

        # 加权缩放 + 融合
        s  = space_features     * gate_weights[:, 0:1]
        td = time_dom_features * gate_weights[:, 1:2]
        tf = time_freq_features * gate_weights[:, 2:3]
        final_features = torch.cat([s, td, tf], dim=1)  # (B, 192)

        # 分类
        output = self.classifier(final_features)

        if return_gate_weights:
            return output, gate_weights
        else:
            return output
