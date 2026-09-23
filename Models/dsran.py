import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.silu(out); out = self.dropout(out)
        out = self.conv2(out); out = self.bn2(out)
        out += self.downsample(identity)
        return self.silu(out)


class ResidualBlockAbtion(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout_rate=0.3):
        super(ResidualBlockAbtion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.silu(out); out = self.dropout(out)
        out = self.conv2(out); out = self.bn2(out)
        out += self.downsample(identity)
        return self.silu(out)


class MACA(nn.Module):
    def __init__(self, in_channels, kernel_size=7, freq=True, time=True, spatial=True):
        super(MACA, self).__init__()
        padding = kernel_size // 2
        self.use_freq = freq
        self.use_time = time
        self.use_spatial = spatial

        if self.use_freq:
            self.conv_freq = DepthwiseSeparableConv(in_channels, in_channels, kernel_size=(kernel_size, 1),
                                                    stride=1, padding=(padding, 0))
        if self.use_time:
            self.conv_time = DepthwiseSeparableConv(in_channels, in_channels, kernel_size=(1, kernel_size),
                                                    stride=1, padding=(0, padding))
        if self.use_spatial:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(in_channels, in_channels // 2),
                nn.SiLU(inplace=True),
                nn.Linear(in_channels // 2, in_channels)
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()
        attn_spatial_explicit = 1.0
        if self.use_freq:
            attn_spatial_explicit *= self.sigmoid(self.conv_freq(x))
        if self.use_time:
            attn_spatial_explicit *= self.sigmoid(self.conv_time(x))

        out = x * attn_spatial_explicit

        if self.use_spatial:
            channel_avg = self.global_pool(x).view(B, C)
            attn_channel = self.fc(channel_avg).view(B, C, 1, 1)
            attn_channel = self.sigmoid(attn_channel)
            out = out * attn_channel

        return out


def get_residual_block_constructor(resblock_abtion_type: str):
    if resblock_abtion_type == "no":
        return ResidualBlock
    elif resblock_abtion_type == "inner abtion":
        return ResidualBlockAbtion
    elif resblock_abtion_type == "replace":
        return lambda in_c, out_c, ks, s, p, dr: nn.Conv2d(in_c, out_c, ks, s, p)
    else:
        raise ValueError(f"Unsupported resblock_abtion_type: {resblock_abtion_type}")


class NetworkStage(nn.Module):
    def __init__(self, block_constructor, in_channels, out_channels, dropout_rate, mac_attention_config):
        super(NetworkStage, self).__init__()
        self.resblock = block_constructor(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=3, stride=1, padding=1, dropout_rate=dropout_rate
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.attn = MACA(out_channels, kernel_size=7, **mac_attention_config)

    def forward(self, x):
        x = self.resblock(x); x = self.pool(x)
        return self.attn(x)


class DSRANet(nn.Module):
    def __init__(self, num_classes, MACA_Attention={'freq': True, 'time': True, 'spatial': True},
                 resblock_abtion_type="no"):
        super(DSRANet, self).__init__()

        channels = [32, 64, 128, 256]
        dropout_rates = [0.5, 0.4, 0.3]

        self.initial_conv = nn.Sequential(
            DepthwiseSeparableConv(12, channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(inplace=True)
        )

        block_constructor = get_residual_block_constructor(resblock_abtion_type)
        self.stages = nn.ModuleList()
        in_c = channels[0]
        for i in range(len(dropout_rates)):
            out_c = channels[i + 1]
            stage = NetworkStage(
                block_constructor=block_constructor,
                in_channels=in_c,
                out_channels=out_c,
                dropout_rate=dropout_rates[i],
                mac_attention_config=MACA_Attention
            )
            self.stages.append(stage)
            in_c = out_c

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(channels[-1], num_classes)
        )

    def forward(self, x):
        x = self.initial_conv(x)
        for stage in self.stages:
            x = stage(x)
        return self.classifier(x)

if __name__ == '__main__':
    import torchinfo

    num_classes = 49
    batch_size = 4
    input_channels = 12
    freq_bins = 65
    time_frames = 19

    model = DSRANet(num_classes=num_classes, resblock_abtion_type="no").eval()
    torchinfo.summary(model, input_size=(batch_size, input_channels, freq_bins, time_frames))