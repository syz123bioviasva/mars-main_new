import torch
from misc.log import log
from dl.vocdataset import VocDataset
from factory.modelfactory import MarsModelFactory
from inference.predictor import DetectionPredictor
from inference.painter import DetectionPainter
from eval.map import MeanAveragePrecision


class ImageEvaluationEntry(object):
    def __init__(self, rawImage, tinfo, truePredBoxes, falsePredBoxes, labelBoxes, truePredClasses, falsePredClasses, labelClasses):
        self.rawImage = rawImage
        self.tinfo = tinfo
        self.truePredBoxes = truePredBoxes
        self.falsePredBoxes = falsePredBoxes
        self.labelBoxes = labelBoxes
        self.truePredClasses = truePredClasses
        self.falsePredClasses = falsePredClasses
        self.labelClasses = labelClasses


class MarsEvaluator(object):
    def __init__(self, mcfg):
        self.mcfg = mcfg
        self.evalDir = self.mcfg.evalDir()

    def initDataLoader(self):
        return VocDataset.getDataLoader(mcfg=self.mcfg, splitName=self.mcfg.testSplitName, isTest=True, fullInfo=True, selectedClasses=self.mcfg.testSelectedClasses)

    def initPredictor(self):
        modelFile = self.mcfg.modelSavePath()
        model = MarsModelFactory.loadPretrainedModel(self.mcfg, modelFile)
        return DetectionPredictor(self.mcfg, model)

    def run(self):
        log.cyan("Mars evaluator running...")

        dataLoader = self.initDataLoader()
        predictor = self.initPredictor()
        batchSize = dataLoader.batch_size
        predList = []
        labelList = []
        tinfoList = []
        rawImageList = []

        for batchIdx, batch in enumerate(dataLoader):
            images, labels, tinfos, rawImages = batch
            images = images.to(self.mcfg.device)
            labels = labels.to(self.mcfg.device)
            preds = predictor.predictRaw(images)
            baseIdx = batchSize * batchIdx
            for imgIdx, pred in enumerate(preds):
                if pred.shape[0] == 0:
                    continue
                expandedPred = torch.zeros(pred.shape[0], pred.shape[1] + 1) # add one column to save image index
                expandedPred[:, 1:] = pred.cpu()
                expandedPred[:, 0] = imgIdx + baseIdx
                predList.append(expandedPred)
            labels[:, 0] += baseIdx
            labels[:, -4:] = labels[:, -4:].mul_(predictor.model.scaleTensor)
            labelList.append(labels)
            tinfoList += tinfos
            rawImageList += rawImages

        prediction = torch.cat(predList, axis=0).cpu()
        groundTruth = torch.cat(labelList, axis=0).cpu()

        mapEvaluator = MeanAveragePrecision(self.mcfg)
        evalRet, truePreds, falsePreds = mapEvaluator.eval(prediction, groundTruth)

        if self.mcfg.paintImages:
            entries = self.classifyByImage(groundTruth, truePreds, falsePreds, tinfoList, rawImageList)
            painter = DetectionPainter(self.mcfg)
            painter.paintImages(entries, self.evalDir)

        return evalRet

    def classifyByImage(self, groundTruth, truePreds, falsePreds, tinfoList, rawImageList):
        entries = []

        for i, (tinfo, rawImage) in enumerate(zip(tinfoList, rawImageList)):
            imgLabelBoxes = []
            imgTruePredBoxes = []
            imgFalsePredBoxes = []
            imgLabelClasses = []
            imgTruePredClasses = []
            imgFalsePredClasses = []

            for j in range(groundTruth.shape[0]):
                if groundTruth[j][0] == i:
                    imgLabelBoxes.append(groundTruth[j][2:])
                    imgLabelClasses.append(groundTruth[j][1])
            for j in range(len(truePreds)):
                if truePreds[j][0] == i:
                    imgTruePredBoxes.append(truePreds[j][3:])
                    imgTruePredClasses.append(truePreds[j][1])
            for j in range(len(falsePreds)):
                if falsePreds[j][0] == i:
                    imgFalsePredBoxes.append(falsePreds[j][3:])
                    imgFalsePredClasses.append(falsePreds[j][1])

            entry = ImageEvaluationEntry(
                rawImage=rawImage,
                tinfo=tinfo,
                truePredBoxes=torch.stack(imgTruePredBoxes) if len(imgTruePredBoxes) > 0 else None,
                falsePredBoxes=torch.stack(imgFalsePredBoxes) if len(imgFalsePredBoxes) > 0 else None,
                labelBoxes=torch.stack(imgLabelBoxes) if len(imgLabelBoxes) > 0 else None,
                truePredClasses=imgTruePredClasses,
                falsePredClasses=imgFalsePredClasses,
                labelClasses=imgLabelClasses,
            )
            entries.append(entry)

        return entries
