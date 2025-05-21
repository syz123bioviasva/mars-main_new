import torch
from misc.bbox import bboxDecode, nonMaxSuppression


class DetectionPredictor(object):
    def __init__(self, mcfg, model):
        self.mcfg = mcfg
        self.model = model
        self.model.setInferenceMode(True)

    def predictRaw(self, images):
        with torch.no_grad():
            preds = self.model(images)

        batchSize = preds[0].shape[0]
        no = self.mcfg.nc + self.mcfg.regMax * 4

        predBoxDistribution, predClassScores = torch.cat([xi.view(batchSize, no, -1) for xi in preds], 2).split((self.mcfg.regMax * 4, self.mcfg.nc), 1)
        predBoxDistribution = predBoxDistribution.permute(0, 2, 1).contiguous() # (batchSize, 8400, regMax * 4)
        predClassScores = predClassScores.sigmoid().permute(0, 2, 1).contiguous() # (batchSize, 8400, nc)

        # generate predicted bboxes
        predBboxes = bboxDecode(self.model.anchorPoints, predBoxDistribution, self.model.proj) # (batchSize, 8400, 4)
        predBboxes = predBboxes * self.model.anchorStrides

        results = nonMaxSuppression(
            predClassScores=predClassScores,
            predBboxes=predBboxes,
            scoreThres=0.2,
            iouThres=0.4,
            maxDetect=50,
        )

        return results
