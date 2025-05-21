import torch
import torch.nn as nn
from model.base.backbone import Bottleneck
class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest'):
        super().__init__()
        self.upsample = nn.Upsample(size=size, scale_factor=scale_factor, mode=mode)

    def forward(self, x):
        return self.upsample(x)

class Neck(nn.Module):
    """
    Reference: resources/yolov8.jpg
    """
    def __init__(self, w, r, n):
        super().__init__()
        self.kernelSize = 3
        self.stride = 2

        # For Y: upsample feat3 (20x20) to 40x40 and merge with feat2 (40x40)
        self.upsample1 = Upsample(scale_factor=2, mode='nearest')
        self.conv_y = nn.Conv2d(int(512 * w * r + 512 * w), int(512 * w), kernel_size=1)
        
        # Y layers: 2 CSP blocks
        self.y_layers = nn.Sequential(
            Bottleneck(int(512 * w), int(512 * w), kernel_size=self.kernelSize),
            Bottleneck(int(512 * w), int(512 * w), kernel_size=self.kernelSize)
        )

        # For X: upsample Y (40x40) to 80x80 and merge with feat1 (80x80)
        self.upsample2 = Upsample(scale_factor=2, mode='nearest')
        self.conv_x = nn.Conv2d(int(512 * w + 256 * w), int(256 * w), kernel_size=1)
        
        # X layers: 2 CSP blocks
        self.x_layers = nn.Sequential(
            Bottleneck(int(256 * w), int(256 * w), kernel_size=self.kernelSize),
            Bottleneck(int(256 * w), int(256 * w), kernel_size=self.kernelSize)
        )

        # For C: downsample X (80x80) to 40x40
        self.conv_c = nn.Conv2d(int(256 * w),int(512 * w), kernel_size=self.kernelSize, stride=self.stride, padding=1)
        
        # C layers: 2 CSP blocks
        self.c_layers = nn.Sequential(
            Bottleneck(int(512 * w), int(512 * w), kernel_size=self.kernelSize),
            Bottleneck(int(512 * w), int(512 * w), kernel_size=self.kernelSize)
        )
        #raise NotImplementedError("Neck::__init__")

    def forward(self, feat1, feat2, feat3):
        """
        Input shape:
            feat1: (B, 256 * w, 80, 80)
            feat2: (B, 512 * w, 40, 40)
            feat3: (B, 512 * w * r, 20, 20)
        Output shape:
            C: (B, 512 * w, 40, 40)
            X: (B, 256 * w, 80, 80)
            Y: (B, 512 * w, 40, 40)
            Z: (B, 512 * w * r, 20, 20)
        """
        # Process Y: upsample feat3 and combine with feat2
        up_feat3 = self.upsample1(feat3)
        concat_y = torch.cat([up_feat3, feat2], dim=1)
        y = self.conv_y(concat_y)
        y = self.y_layers(y)

        # Process X: upsample Y and combine with feat1
        up_y = self.upsample2(y)
        concat_x = torch.cat([up_y, feat1], dim=1)
        x = self.conv_x(concat_x)
        x = self.x_layers(x)

        # Process C: downsample X to match feat2's scale
        c = self.conv_c(x)
        c = self.c_layers(c)

        # Z is directly taken from feat3
        z = feat3

        return c, x, y, z
        #raise NotImplementedError("Neck::forward")
