import os
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from functools import partial
import pathlib
import numpy as np
from functools import partial
from misc.log import log
from misc import img, xml
from dl.aug import DataAugmentationProcessor


class VocDataset(Dataset):
    @staticmethod
    def collate(batch):
        """
        Used by PyTorch DataLoader class (collate_fn)
        """
        images  = []
        labels  = []
        tinfos = []
        rawImages = []

        for i, data in enumerate(batch):
            img = data[0]
            label = data[1]
            images.append(img)
            label[:, 0] = i # enrich image index in batch
            labels.append(label)
            if len(data) > 3:
                tinfo = data[2]
                tinfos.append(tinfo)
                rawImage = data[3]
                rawImages.append(rawImage)

        images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
        labels  = torch.from_numpy(np.concatenate(labels, 0)).type(torch.FloatTensor)

        if len(rawImages) > 0:
            return images, labels, tinfos, rawImages

        return images, labels

    @staticmethod
    def workerInit(seed, workerId):
        workerSeed = workerId + seed
        random.seed(workerSeed)
        np.random.seed(workerSeed)
        torch.manual_seed(workerSeed)

    @staticmethod
    def getDataLoader(mcfg, splitName, isTest, fullInfo, selectedClasses=None):
        if splitName not in mcfg.subsetMap:
            raise ValueError("Split not found in mcfg: {}".format(splitName))

        dataset = VocDataset(
            imageDir=mcfg.imageDir,
            annotationDir=mcfg.annotationDir,
            classList=mcfg.classList,
            inputShape=mcfg.inputShape,
            subset=mcfg.subsetMap[splitName],
            isTest=isTest,
            fullInfo=fullInfo,
            suffix=mcfg.suffix,
            splitName=splitName,
            selectedClasses=selectedClasses,
        )
        return DataLoader(
            dataset,
            shuffle=True,
            batch_size=mcfg.batchSize,
            num_workers=mcfg.dcore,
            pin_memory=True,
            drop_last=False,
            sampler=None,
            collate_fn=VocDataset.collate,
            worker_init_fn=partial(VocDataset.workerInit, mcfg.seed)
        )

    def __init__(self, imageDir, annotationDir, classList, inputShape, subset, isTest, fullInfo, suffix, splitName, selectedClasses):
        super(VocDataset, self).__init__()
        self.imageDir = imageDir
        self.annotationDir = annotationDir
        self.classList = classList
        self.inputShape = inputShape
        self.augp = DataAugmentationProcessor(
            inputShape=inputShape,
            rescalef=(0.4, 1.8),
            huef=0.2,
            satf=0.8,
            valf=0.5
        )
        self.isTest = isTest
        self.fullInfo = fullInfo
        self.suffix = suffix
        self.splitName = splitName
        self.selectedClasses = selectedClasses

        if subset is None:
            self.imageFiles = [os.path.join(imageDir, x) for x in os.listdir(imageDir) if pathlib.Path(x).suffix == self.suffix]
        else:
            self.imageFiles = [os.path.join(imageDir, x) for x in subset]
            for imFile in self.imageFiles:
                if not os.path.exists(imFile):
                    raise ValueError("Image file in subset not exists: {}".format(imFile))
        if len(self.imageFiles) == 0:
            raise ValueError("Empty image directory: {}".format(imageDir))

        self.annotationFiles = [os.path.join(annotationDir, "{}.xml".format(pathlib.Path(x).stem)) for x in self.imageFiles]
        for annFile in self.annotationFiles:
            if not os.path.exists(annFile):
                raise ValueError("Annotation file not exists: {}".format(annFile))

        log.inf("VOC dataset [{}] initialized from {} with {} images".format(self.splitName, imageDir, len(self.imageFiles)))
        if self.selectedClasses is not None:
            log.inf("VOC dataset [{}] set with selected classes: {}".format(self.splitName, self.selectedClasses))

    def postprocess(self, imageData, boxList):
        imageData = imageData / 255.0
        imageData = np.transpose(np.array(imageData, dtype=np.float32), (2, 0, 1))
        boxList = np.array(boxList, dtype=np.float32)
        labels = np.zeros((boxList.shape[0], 6)) # add one dim (5 + 1 = 6) as image batch index (VocDataset.collate)
        if boxList.shape[0] > 0:
            boxList[:, [0, 2]] = boxList[:, [0, 2]] / self.inputShape[1]
            boxList[:, [1, 3]] = boxList[:, [1, 3]] / self.inputShape[0]
            labels[:, 1] = boxList[:, -1]
            labels[:, 2:] = boxList[:, :4]
        return imageData, labels

    def __len__(self):
        return len(self.imageFiles)

    def __getitem__(self, index):
        ii = index % len(self.imageFiles)
        imgFile = self.imageFiles[ii]
        image = img.loadRGBImage(imgFile)
        annFile = self.annotationFiles[ii]
        boxList = xml.XmlBbox.loadXmlObjectList(annFile, self.classList, selectedClasses=self.selectedClasses, asArray=True)

        if self.isTest:
            imageData, boxList, tinfo = self.augp.processSimple(image, boxList)
        else:
            imageData, boxList, tinfo = self.augp.processEnhancement(image, boxList)

        imageData, labels = self.postprocess(imageData, boxList)
        if not self.fullInfo:
            return imageData, labels

        tinfo.imgFile = imgFile
        return imageData, labels, tinfo, image
