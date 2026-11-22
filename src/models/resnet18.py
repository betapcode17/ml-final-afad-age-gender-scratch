# Kiến trúc ResNet-18 Backbone (Train từ đầu - KHÔNG dùng pretrained)
# Hỗ trợ:
# 1. ResNet18Backbone (use_se=False: chuẩn | use_se=True: + SE-Block Attention)
# 2. ResNet18CustomBackbone (ResNet-D Deep Stem 3x3 + AvgPool Shortcut + CBAM [Channel+Spatial] + Multi-Scale layer3+layer4)

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Channel Attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.excite(self.squeeze(x).view(b, c)).view(b, c, 1, 1)
        return x * w


class ChannelAttention(nn.Module):
    """Nhánh Channel Attention của CBAM (kết hợp AvgPool + MaxPool)."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))


class SpatialAttention(nn.Module):
    """Nhánh Spatial Attention của CBAM (tập trung vào mắt, mũi, miệng, đường hàm; bỏ nhiễu nền)."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        att = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * att


class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        return self.sa(self.ca(x))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, use_se=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        return self.relu(out)


class BasicBlockD_CBAM(nn.Module):
    """Residual Block cải tiến chuẩn ResNet-D (AvgPool shortcut không mất pixel) + CBAM Attention."""

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.cbam = CBAMBlock(out_channels, reduction=16, kernel_size=7)

        if stride != 1 or in_channels != self.expansion * out_channels:
            layers = []
            if stride != 1:
                layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True))
            layers.extend([
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels),
            ])
            self.shortcut = nn.Sequential(*layers)
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.cbam(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        return self.relu(out)


class ResNet18Backbone(nn.Module):
    def __init__(self, in_channels=3, use_se=False):
        super().__init__()
        self.in_channels = 64
        self.use_se = use_se
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_features = 512 * BasicBlock.expansion
        self._init_weights()

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride=s, use_se=self.use_se))
            self.in_channels = out_channels * BasicBlock.expansion
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.avgpool(x).flatten(1)


class ResNet18CustomBackbone(nn.Module):
    """Kiến trúc nâng cấp 02d (Train From Scratch):
    - Deep Stem 3x3x3 (giữ chi tiết da & nếp nhăn mảnh hơn Conv 7x7)
    - ResNet-D AvgPool Shortcut + CBAM (Channel + Spatial Attention)
    - Multi-Scale Fusion: feat4 (512-d) cho Gender & feat34 = cat([feat3, feat4]) (768-d) cho Age
    """

    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_features = 512
        self.out_age_features = 256 + 512  # Multi-scale layer3 (14x14) + layer4 (7x7)
        self._init_weights()

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlockD_CBAM(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels * BasicBlockD_CBAM.expansion
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_multiscale(self, x):
        x = self.maxpool(self.stem(x))
        x = self.layer1(x)
        x = self.layer2(x)
        f3 = self.layer3(x)
        f4 = self.layer4(f3)
        feat3 = self.avgpool(f3).flatten(1)  # (B, 256)
        feat4 = self.avgpool(f4).flatten(1)  # (B, 512)
        return feat4, torch.cat([feat3, feat4], dim=1)  # (B, 512), (B, 768)

    def forward(self, x):
        _, feat34 = self.forward_multiscale(x)
        return feat34
