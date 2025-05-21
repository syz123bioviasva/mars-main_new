import torch
import pandas as pd
from misc.bbox import iou


class MeanAveragePrecision(object):
    def __init__(self, mcfg):
        self.mcfg = mcfg
        self.minIou = mcfg.minIou
        self.epsilon = 1e-6

    def eval(self, prediction, groundTruth):
        APs = torch.zeros(self.mcfg.nc)
        prediction = prediction.cpu()
        groundTruth = groundTruth.cpu()
        truePreds = []
        falsePreds = []

        for classIdx in range(self.mcfg.nc):
            AP, classTruePreds, classFalsePreds = self.evalClass(prediction, groundTruth, classIdx)
            APs[classIdx] = AP
            truePreds += classTruePreds
            falsePreds += classFalsePreds

        evalRet = pd.DataFrame({
            "classIndex": [x for x in range(self.mcfg.nc)],
            "className": self.mcfg.classList,
            "AP": APs,
        })

        return evalRet, truePreds, falsePreds

    def evalClass(self, prediction, groundTruth, classIdx):
        classPreds = []
        classTruePreds = []
        classFalsePreds = []
        classTruths = []

        for imageIdx in range(prediction.shape[0]):
            if prediction[imageIdx][1] == classIdx:
                classPreds.append(prediction[imageIdx])

        for imageIdx in range(groundTruth.shape[0]):
            if groundTruth[imageIdx][1] == classIdx:
                classTruths.append(groundTruth[imageIdx])

        if len(classTruths) == 0:
            return torch.nan, classPreds, []

        TP = torch.zeros(len(classPreds))
        FP = torch.zeros(len(classPreds))
        usedImages = set()

        for i, classPred in enumerate(classPreds):
            matchedTruths = []
            matchedIndexes = []
            for j, x in enumerate(classTruths):
                if x[0] == classPred[0]:
                    matchedTruths.append(x)
                    matchedIndexes.append(j)

            bestIou = 0; bestIdx = None
            for idx, matchedTruth in enumerate(matchedTruths):
                matchedIou = iou(classPred[3:], matchedTruth[2:]).squeeze()
                if matchedIou > bestIou:
                    bestIou = matchedIou
                    bestIdx = matchedIndexes[idx]

            if bestIou > self.minIou and bestIdx not in usedImages:
                TP[i] = 1
                classTruePreds.append(classPred)
            else:
                FP[i] = 1
                classFalsePreds.append(classPred)

        TPcs = TP.cumsum(dim=0)
        FPcs = FP.cumsum(dim=0)
        recalls = TPcs / (len(classTruths) + self.epsilon)
        precisions = TPcs.div(TPcs + FPcs + self.epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        AP = torch.trapz(precisions, recalls)

        return AP, classTruePreds, classFalsePreds
