import os
import sys
from sklearn.model_selection import train_test_split
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from misc.xml import XmlBbox
import pandas as pd


class Mar20DataSpliter(object):
    def __init__(self):
        self.trainRatio = 0.94
        self.valRatio = 0.03
        self.testRatio = 0.03
        self.smallRatio = 0.01
        self.classList = ["A{}".format(x) for x in range(1, 21)]

        self.imageDir = "/auto/cvdata/mar20/images"
        self.annotationDir = "/auto/cvdata/mar20/annotations"
        self.splitDir = "/auto/cvdata/mar20/splits/v5"
        if not os.path.exists(self.splitDir):
            os.makedirs(self.splitDir)

        self.imageFiles = [x for x in os.listdir(self.imageDir) if x.endswith(".jpg")]

    def getAnnFile(self, imageFile):
        baseName = os.path.basename(imageFile).split(".")[0]
        annFile = os.path.join(self.annotationDir, "{}.xml".format(baseName))
        return annFile

    def loadImageSetClassCount(self, imageFiles):
        classCount = {}
        classImages = {}

        for imgFile in imageFiles:
            annFile = self.getAnnFile(imgFile)
            objList = XmlBbox.loadXmlObjectList(annFile, self.classList, asArray=False)
            for obj in objList:
                if obj.className in classCount:
                    classCount[obj.className] += 1
                    classImages[obj.className].append(imgFile)
                else:
                    classCount[obj.className] = 1
                    classImages[obj.className] = [imgFile]

        classNames = []
        counts = []

        for className, count in classCount.items():
            classNames.append(className)
            counts.append(count)

        cdf = pd.DataFrame({
            "className": classNames,
            "count": counts,
        })
        return cdf, classImages

    def genSmallSet(self, trainImages):
        totalDf, classImages = self.loadImageSetClassCount(trainImages)

        usedImages = set()
        smallSet = set()

        for className, subFiles in classImages.items():
            _, small = train_test_split(subFiles, test_size=self.smallRatio)

            for img in small:
                if img in usedImages:
                    continue
                smallSet.add(img)
                usedImages.add(img)

        smallDf, _ = self.loadImageSetClassCount(smallSet)
        return smallDf, smallSet

    def genSets(self):
        totalDf, classImages = self.loadImageSetClassCount(self.imageFiles)

        usedImages = set()
        testSet = set()
        valSet = set()
        trainSet = set()

        for className, subFiles in classImages.items():
            train, testVal = train_test_split(subFiles, test_size=self.testRatio + self.valRatio)
            test, val = train_test_split(testVal, test_size=self.valRatio / (self.testRatio + self.valRatio))

            for img in test:
                if img in usedImages:
                    continue
                testSet.add(img)
                usedImages.add(img)

            for img in val:
                if img in usedImages:
                    continue
                valSet.add(img)
                usedImages.add(img)

            for img in train:
                if img in usedImages:
                    continue
                trainSet.add(img)
                usedImages.add(img)

        testDf, _ = self.loadImageSetClassCount(testSet)
        testDf = testDf.rename(columns={"count": "testSize"})

        valDf, _ = self.loadImageSetClassCount(valSet)
        valDf = valDf.rename(columns={"count": "valSize"})

        trainDf, _ = self.loadImageSetClassCount(trainSet)
        trainDf = trainDf.rename(columns={"count": "trainSize"})

        smallDf, smallSet = self.genSmallSet(trainSet)
        smallDf = smallDf.rename(columns={"count": "smallCount"})

        cdf = totalDf.merge(testDf, on="className", how="outer")
        cdf = cdf.merge(valDf, on="className", how="outer")
        cdf = cdf.merge(trainDf, on="className", how="outer")
        cdf = cdf.merge(smallDf, on="className", how="outer")

        with open(os.path.join(self.splitDir, "train.txt"), "w") as f:
            f.writelines(["{}\n".format(x) for x in trainSet])
        with open(os.path.join(self.splitDir, "test.txt"), "w") as f:
            f.writelines(["{}\n".format(x) for x in testSet])
        with open(os.path.join(self.splitDir, "validation.txt"), "w") as f:
            f.writelines(["{}\n".format(x) for x in valSet])
        with open(os.path.join(self.splitDir, "small.txt"), "w") as f:
            f.writelines(["{}\n".format(x) for x in smallSet])

        cdfFile = os.path.join(self.splitDir, "cdf.csv")
        cdf.to_csv(cdfFile, index=False, na_rep="nan")



if __name__ == "__main__":
    processor = Mar20DataSpliter()
    processor.genSets()

# restImages = [x for x in imageFiles if x not in excludeded]

# # split rest images

# with open(os.path.join(splitDir, "train.txt"), "w") as f:
#     f.writelines(["{}\n".format(x) for x in train])

# with open(os.path.join(splitDir, "test.txt"), "w") as f:
#     f.writelines(["{}\n".format(x) for x in test])

# with open(os.path.join(splitDir, "validation.txt"), "w") as f:
#     f.writelines(["{}\n".format(x) for x in val])

# # split excluded images
# etran, etest = train_test_split(excludeded, test_size=0.2)

# with open(os.path.join(splitDir, "extrain.txt"), "w") as f:
#     f.writelines(["{}\n".format(x) for x in etran])

# with open(os.path.join(splitDir, "extest.txt"), "w") as f:
#     f.writelines(["{}\n".format(x) for x in etest])

# with open(os.path.join(splitDir, "ex.txt"), "w") as f:
#     f.writelines(["{}\n".format(x) for x in excludeded])
