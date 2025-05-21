import torch.nn as nn
import torch.nn.functional as F
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential() if in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        identity = self.downsample(identity)
        x += identity
        return self.relu(x)


class Backbone(nn.Module):
    """
    Reference: resources/yolov8.jpg
    """
    def __init__(self, w, r, n):
        super().__init__()
        self.imageChannel = 3
        self.kernelSize = 3
        self.stride = 2

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(self.imageChannel, int(64 * w), kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(int(64 * w)),
            nn.ReLU(inplace=True)
        )

        # Stage 1: stride 2
        self.stage1 = nn.Sequential(
            Bottleneck(int(64 * w), int(128 * w), kernel_size=self.kernelSize, stride=self.stride),
            *[Bottleneck(int(128 * w), int(128 * w), kernel_size=self.kernelSize) for _ in range(n-1)]
        )

        # Stage 2: stride 2
        self.stage2 = nn.Sequential(
            Bottleneck(int(128 * w), int(256 * w), kernel_size=self.kernelSize, stride=self.stride),
            *[Bottleneck(int(256 * w), int(256 * w), kernel_size=self.kernelSize) for _ in range(n-1)]
        )

        # Stage 3: stride 2
        self.stage3 = nn.Sequential(
            Bottleneck(int(256 * w), int(512 * w), kernel_size=self.kernelSize, stride=self.stride),
            *[Bottleneck(int(512 * w), int(512 * w), kernel_size=self.kernelSize) for _ in range(n-1)]
        )

        # Stage 4: stride 2 (with ratio applied to channels)
        self.stage4 = nn.Sequential(
            Bottleneck(int(512 * w), int(512 * w * r), kernel_size=self.kernelSize, stride=self.stride),
            *[Bottleneck(int(512 * w * r), int(512 * w * r), kernel_size=self.kernelSize) for _ in range(n-1)]
        )

    def forward(self, x):
        x = self.stem(x)
        
        # Stage1 (output: 160x160)
        feat0 = self.stage1(x)
        
        # Stage2 (output: 80x80)
        feat1 = self.stage2(feat0)
        
        # Stage3 (output: 40x40)
        feat2 = self.stage3(feat1)
        
        # Stage4 (output: 20x20)
        feat3 = self.stage4(feat2)
        
        return feat0, feat1, feat2, feat3