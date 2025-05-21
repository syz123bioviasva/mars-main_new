import torch
import torch.nn as nn
from misc.log import log
from overrides import override # this could be removed since Python 3.12
from model.base.yolomodel import YoloModel
from model.distillation.teachermodel import YoloTeacherModel
from model.base.backbone import Backbone
from model.base.neck import Neck
from model.base.head import DetectHead as Head


class YoloStudentModel(YoloModel):
    def __init__(self, mcfg):
        super().__init__(mcfg)
        self.mcfg = mcfg
        self.teacherModel = self.initTeacherModel()
        
        # 特征蒸馏层
        self.distill_layers = nn.ModuleList([
            nn.Conv2d(256, 256, 1),  # P3
            nn.Conv2d(512, 512, 1),  # P4
            nn.Conv2d(1024, 1024, 1) # P5
        ])

    def initTeacherModel(self):
        return YoloTeacherModel.loadModelFromFile(self.mcfg, self.mcfg.teacherModelFile)

    def getTrainLoss(self):
        from train.distilloss import DistillationDetectionLoss
        return DistillationDetectionLoss(self.mcfg, self)

    @override
    def forward(self, x):
        if self.inferenceMode:
            with torch.no_grad():
                feat0, feat1, feat2, feat3 = self.backbone.forward(x)
                C, X, Y, Z = self.neck.forward(feat1, feat2, feat3)
                xo, yo, zo = self.head.forward(X, Y, Z)
                return xo, yo, zo

        feat0, feat1, feat2, feat3 = self.backbone.forward(x)
        C, X, Y, Z = self.neck.forward(feat1, feat2, feat3)
        xo, yo, zo = self.head.forward(X, Y, Z)
        tlayerOutput = self.teacherModel.forward(x)
        return (xo, yo, zo, feat0, feat1, feat2, feat3, C, X, Y, Z), tlayerOutput

    @override
    def load(self, modelFile):
        """
        Load states except "self.teacherModel"
        """
        selfState = self.state_dict()
        loadedState = torch.load(modelFile, weights_only=True)
        selfState.update(loadedState)
        missingKeys, unexpectedKeys = self.load_state_dict(selfState, strict=False)
        if len(unexpectedKeys) > 0:
            log.yellow("Unexpected keys found in model file, ignored:\nunexpected={}\nurl={}".format(unexpectedKeys, modelFile))
        if len(missingKeys) > 0:
            log.red("Missing keys in model file:\nmissing={}\nurl={}".format(missingKeys, modelFile))
            import pdb; pdb.set_trace()
        else:
            log.grey("Yolo student model loaded from file: {}".format(modelFile))
