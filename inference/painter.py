import os
import cv2
import numpy as np
from misc.log import log
from misc.bbox import recoverBoxes, isValidBox


class DetectionPainter(object):
    def __init__(self, mcfg):
        self.mcfg = mcfg
        self.labelColor = (255, 0, 255) # purple in BGR
        self.truePredColor = (0, 255, 0) # green
        self.falsePredColor = (0, 255, 255) # yellow
        self.boxThickness = 2
        self.textColor = (255, 255, 255)
        self.textThickness = 1
        self.textScale = 0.5
        self.textFont = cv2.FONT_HERSHEY_SIMPLEX

    def calcRecoveredBoxes(self, boxes, tinfo):
        return recoverBoxes(
            boxes,
            oriWidth=tinfo.oriWidth,
            oriHeight=tinfo.oriHeight,
            scaledWidth=tinfo.scaledWidth,
            scaledHeight=tinfo.scaledHeight,
            targetWidth=tinfo.targetWidth,
            targetHeight=tinfo.targetHeight,
            xoffset=tinfo.xoffset,
            yoffset=tinfo.yoffset,
            flip=tinfo.flip,
        )

    def paintBoxWithText(self, cvImg, box, classIndex, color):
        if not isValidBox(box):
            return
        labelText = self.mcfg.classList[int(classIndex)]
        labelSize = cv2.getTextSize(labelText, self.textFont, self.textScale, self.textThickness)[0]
        # paint box
        x0, y0, x1, y1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(cvImg, (x0, y0), (x1, y1), color=color, thickness=self.boxThickness)
        # paint text and background box
        if y0 - labelSize[1] - 3 < 0:
            textBoxStart = (x0, y0 + 3)
            textBoxEnd = (x0 + labelSize[0], y0 + labelSize[1] + 3)
            textXY = (x0, y0 + labelSize[1] + 3)
        else:
            textBoxStart = (x0, y0 - labelSize[1] - 3)
            textBoxEnd = (x0 + labelSize[0], y0 - 3)
            textXY = (x0, y0 - 3)
            cv2.rectangle(cvImg, textBoxStart, textBoxEnd, color=color, thickness=-1)
            cv2.putText(cvImg, labelText, textXY, self.textFont, self.textScale, self.textColor, thickness=self.textThickness)

    def paintImage(self, entry):
        cvImg = cv2.cvtColor(np.array(entry.rawImage), cv2.COLOR_RGB2BGR)
        # paint ground truth
        if entry.labelBoxes is not None:
            boxes = self.calcRecoveredBoxes(entry.labelBoxes, entry.tinfo)
            for i in range(boxes.shape[0]):
                self.paintBoxWithText(cvImg, boxes[i], entry.labelClasses[i], color=self.labelColor)
        # paint true preds
        if entry.truePredBoxes is not None:
            boxes = self.calcRecoveredBoxes(entry.truePredBoxes, entry.tinfo)
            for i in range(boxes.shape[0]):
                self.paintBoxWithText(cvImg, boxes[i], entry.truePredClasses[i], color=self.truePredColor)
        # paint false preds
        if entry.falsePredBoxes is not None:
            boxes = self.calcRecoveredBoxes(entry.falsePredBoxes, entry.tinfo)
            for i in range(boxes.shape[0]):
                self.paintBoxWithText(cvImg, boxes[i], entry.falsePredClasses[i], color=self.falsePredColor)
        return cvImg

    def paintImages(self, entries, outputDir):
        for entry in entries:
            cvImg = self.paintImage(entry)
            baseName = os.path.basename(entry.tinfo.imgFile)
            outputFile = os.path.join(outputDir, baseName)
            cv2.imwrite(outputFile, cvImg)
        log.inf("Painted images of {} saved at {}".format(len(entries), outputDir))
