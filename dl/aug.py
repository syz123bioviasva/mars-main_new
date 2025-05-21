import numpy as np
from PIL import Image
from misc import misc
from misc.img import rescale2Target, hsvAdjust
from misc.bbox import rescaleBoxes
import random
import cv2
from misc.log import log
import math


class ImageTransformedInfo(object):
    def __init__(self, oriWidth, oriHeight, scaledWidth, scaledHeight, targetWidth, targetHeight, xoffset, yoffset, flip):
        self.oriWidth = oriWidth
        self.oriHeight = oriHeight
        self.scaledWidth = scaledWidth
        self.scaledHeight = scaledHeight
        self.targetWidth = targetWidth
        self.targetHeight = targetHeight
        self.xoffset = xoffset
        self.yoffset = yoffset
        self.flip = flip
        self.imgFile = None


class DataAugmentationProcessor(object):
    def __init__(self, inputShape, jitter=0.3, rescalef=(0.4, 1.8), flipProb=0.5, huef=0.015, satf=0.7, valf=0.4):
        self.inputShape = inputShape
        self.jitter = jitter
        self.rescalef = rescalef
        self.flipProb = flipProb
        self.huef = huef
        self.satf = satf
        self.valf = valf
        self.rotateProb = 0.5  # 随机旋转的概率
        self.rotateRange = (-10, 10)  # 旋转角度范围

    def processEnhancement(self, image, boxList):
        # 随机旋转
        if random.random() < self.rotateProb:
            image, boxList = self._random_rotate(image, boxList)
        return self.processSimple(image, boxList)

    def _random_rotate(self, image, boxList):
        """随机旋转图像和边界框"""
        angle = random.uniform(self.rotateRange[0], self.rotateRange[1])  # 旋转角度范围
        w, h = image.size
        
        # 旋转图像
        image = image.rotate(angle, expand=True)
        
        # 调整边界框
        if len(boxList) > 0:
            # 将边界框转换为中心点坐标
            boxes = boxList.copy()
            boxes[:, 0] = (boxList[:, 0] + boxList[:, 2]) / 2
            boxes[:, 1] = (boxList[:, 1] + boxList[:, 3]) / 2
            boxes[:, 2] = boxList[:, 2] - boxList[:, 0]
            boxes[:, 3] = boxList[:, 3] - boxList[:, 1]
            
            # 旋转边界框
            angle_rad = math.radians(angle)
            cos_angle = math.cos(angle_rad)
            sin_angle = math.sin(angle_rad)
            
            # 计算旋转后的坐标
            boxes[:, 0] = boxes[:, 0] * cos_angle - boxes[:, 1] * sin_angle
            boxes[:, 1] = boxes[:, 0] * sin_angle + boxes[:, 1] * cos_angle
            
            # 转回左上右下坐标
            boxList[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
            boxList[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
            boxList[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
            boxList[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
            
            # 确保坐标在有效范围内
            boxList[:, 0] = np.clip(boxList[:, 0], 0, w)
            boxList[:, 1] = np.clip(boxList[:, 1], 0, h)
            boxList[:, 2] = np.clip(boxList[:, 2], 0, w)
            boxList[:, 3] = np.clip(boxList[:, 3], 0, h)
            
            # 过滤掉无效的边界框
            valid_boxes = []
            for box in boxList:
                if box[2] > box[0] and box[3] > box[1]:  # 确保宽高为正
                    valid_boxes.append(box)
            boxList = np.array(valid_boxes) if valid_boxes else boxList
        
        return image, boxList

    def processSimple(self, image, boxList):
        # rescale image
        targetHeight, targetWidth = self.inputShape
        oriWidth, oriHeight = image.size
        scaleFactor = min(targetWidth / oriWidth, targetHeight / oriHeight)
        scaledWidth = int(oriWidth * scaleFactor)
        scaledHeight = int(oriHeight * scaleFactor)
        newImage, xoffset, yoffset = rescale2Target(image, scaledWidth, scaledHeight, targetWidth, targetHeight)
        imageData = np.array(newImage, np.float32)
        # rescale boxes accordingly
        boxList = rescaleBoxes(boxList, oriWidth, oriHeight, scaledWidth, scaledHeight, targetWidth, targetHeight, xoffset, yoffset, False)
        transformInfo = ImageTransformedInfo(oriWidth, oriHeight, scaledWidth, scaledHeight, targetWidth, targetHeight, xoffset, yoffset, False)
        return imageData, boxList, transformInfo
