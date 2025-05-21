import torch
from model.base.yolomodel import YoloModel
from overrides import override # this could be removed since Python 3.12


class YoloTeacherModel(YoloModel):
    def __init__(self, mcfg):
        super().__init__(mcfg)
        self.eval()

    @override
    def forward(self, x):
        with torch.no_grad():
            feat0, feat1, feat2, feat3 = self.backbone.forward(x)
            C, X, Y, Z = self.neck.forward(feat1, feat2, feat3)
            xo, yo, zo = self.head.forward(X, Y, Z)
        return xo, yo, zo, feat0, feat1, feat2, feat3, C, X, Y, Z
