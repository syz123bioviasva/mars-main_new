import os
import sys
import pandas as pd
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from misc.xml import XmlBbox


annDir = "/auto/cvdata/mar20/annotations"
classList = ["A{}".format(x) for x in range(1, 21)]


def checkIntersection(splitFile):
    testSplit = "/auto/cvdata/mar20/splits/v4/test.txt"
    with open(testSplit) as f:
        testTokens = [x.strip() for x in f.readlines()]
        testTokens = [x for x in testTokens if len(x) > 0]
        testSet = set(testTokens)

    with open(splitFile) as f:
        tokens = [x.strip() for x in f.readlines()]
        tokens = [x for x in tokens if len(x) > 0]

    inter = testSet.intersection(set(tokens))
    print("Intersection with test: split={},inter={}".format(splitFile, len(inter)))


def load(splitFile):
    classObjCount = {}
    classImgCount = {}
    with open(splitFile) as f:
        tokens = [x.strip() for x in f.readlines()]
        tokens = [x for x in tokens if len(x) > 0]
        for img in tokens:
            baseName = img.split(".")[0]
            annFile = os.path.join(annDir, "{}.xml".format(baseName))
            objList = XmlBbox.loadXmlObjectList(annFile, classList, asArray=False)
            clsSet = set()
            for obj in objList:
                clsSet.add(obj.className)
                if obj.className in classObjCount:
                    classObjCount[obj.className] += 1
                else:
                    classObjCount[obj.className] = 1
            for cls in clsSet:
                if cls in classImgCount:
                    classImgCount[cls] += 1
                else:
                    classImgCount[cls] = 1

    classes = []
    objCounts = []
    imgCounts = []
    classIndexes = []
    for cls, count in classObjCount.items():
        classIndexes.append(classList.index(cls))
        classes.append(cls)
        objCounts.append(count)
        if cls in classImgCount:
            imgCounts.append(classImgCount[cls])
        else:
            imgCounts.append(0)

    df = pd.DataFrame(data={"index": classIndexes, "class": classes, "objCount": objCounts, "imgCounts": imgCounts}).sort_values(by="index")
    print("Eval result for: {}\n{}\n".format(splitFile, df))


splitFileList = [
    # "/auto/cvdata/mar20/splits/v3/c10new.txt",
    # "/auto/cvdata/mar20/splits/v3/test.txt",
    "/auto/cvdata/mar20/splits/v4/train.txt",
    "/auto/cvdata/mar20/splits/v4/random.txt",
]

for file in splitFileList:
    checkIntersection(file)

for file in splitFileList:
    load(file)


