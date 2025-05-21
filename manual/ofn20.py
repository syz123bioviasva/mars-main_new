import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from misc.xml import XmlBbox


splitsDir = "/auto/cvdata/mar20/splits/v3"

trainFile = os.path.join(splitsDir, "train.txt")
with open(trainFile) as f:
    trainImages = [x.strip() for x in f.readlines()]
    trainImages = [x for x in trainImages if len(x) > 0]

classList = ["A{}".format(x) for x in range(1, 21)]
annDir = "/auto/cvdata/mar20/annotations"

c10File = os.path.join(splitsDir, "c10.txt")
with open(c10File) as f:
    c10Images = [x.strip() for x in f.readlines()]
    c10Images = [x for x in c10Images if len(x) > 0]

filteredImages = []
for img in trainImages:
    baseName = img.split(".")[0]
    annFile = os.path.join(annDir, "{}.xml".format(baseName))
    objList = XmlBbox.loadXmlObjectList(annFile, classList, asArray=False)
    imgClasses = []
    excluded = False
    for obj in objList:
        if obj.className in ["A1", "A2"]:
            excluded = True
            break
    if not excluded:
        filteredImages.append(img)

print(len(filteredImages))




for c10Img in c10Images:
    if c10Img in filteredImages:
        continue
    baseName = img.split(".")[0]
    annFile = os.path.join(annDir, "{}.xml".format(baseName))
    objList = XmlBbox.loadXmlObjectList(annFile, classList, asArray=False)
    filteredImages.append(c10Img)

outputFile = os.path.join(splitsDir, "ofn20.txt")
with open(outputFile, "w") as f:
    f.writelines(["{}\n".format(x) for x in filteredImages])

import pdb; pdb.set_trace()
