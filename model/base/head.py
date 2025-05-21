import torch
import torch.nn as nn
from model.base.components import Conv


class DetectHead(nn.Module):
    """
    Reference: resources/yolov8.jpg
    """
    def __init__(self, w, r, nc, regMax):
        super().__init__()
        self.kernelSize = 3
        self.stride = 1
        self.nc = nc
        self.regMax = regMax
        self.ch = (int(256 * w), int(512 * w), int(512 * w * r))

        c2 = max((16, self.ch[0] // 4, self.regMax * 4))
        c3 = max(self.ch[0], nc)  # channels

        self.bboxHead = nn.ModuleList(nn.Sequential(Conv(x, c2, self.kernelSize, self.stride), Conv(c2, c2, self.kernelSize, self.stride), nn.Conv2d(c2, 4 * self.regMax, 1)) for x in self.ch)
        self.classifyHead = nn.ModuleList(nn.Sequential(Conv(x, c3, self.kernelSize, self.stride), Conv(c3, c3, self.kernelSize, self.stride), nn.Conv2d(c3, self.nc, 1)) for x in self.ch)

    def forward(self, X, Y, Z):
        """
        Input shape:
            X: (B, 256 * w, 80, 80)
            Y: (B, 512 * w, 40, 40)
            Z: (B, 512 * w * r, 20, 20)
        Output shape:
            xo: (B, regMax * 4 + nc, 80, 80)
            yo: (B, regMax * 4 + nc, 40, 40)
            zo: (B, regMax * 4 + nc, 20, 20)
        """

        xo = torch.cat((self.bboxHead[0](X), self.classifyHead[0](X)), 1)
        yo = torch.cat((self.bboxHead[1](Y), self.classifyHead[1](Y)), 1)
        zo = torch.cat((self.bboxHead[2](Z), self.classifyHead[2](Z)), 1)

        return xo, yo, zo
