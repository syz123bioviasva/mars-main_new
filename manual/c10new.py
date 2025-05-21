import os
import sys
import random
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from misc.xml import XmlBbox


splitsDir = "/auto/cvdata/mar20/splits/v3"
classList = ["A{}".format(x) for x in range(1, 21)]
annDir = "/auto/cvdata/mar20/annotations"


trainFile = os.path.join(splitsDir, "train.txt")
with open(trainFile) as f:
    trainImages = [x.strip() for x in f.readlines()]
    trainImages = [x for x in trainImages if len(x) > 0]

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

c10File = os.path.join(splitsDir, "c10.txt")
with open(c10File) as f:
    c10Images = [x.strip() for x in f.readlines()]
    c10Images = [x for x in c10Images if len(x) > 0]

A1Only = []
A2Only = []
Rest = []

for c10Img in c10Images:
    baseName = c10Img.split(".")[0]
    annFile = os.path.join(annDir, "{}.xml".format(baseName))
    objList = XmlBbox.loadXmlObjectList(annFile, classList, asArray=False)
    hasA1 = False; hasA2 = False
    for obj in objList:
        if obj.className == "A1":
            hasA1 = True
        elif obj.className == "A2":
            hasA2 = True
    if hasA2 and hasA1:
        continue
    if hasA2:
        A2Only.append(c10Img)
    elif hasA1:
        A1Only.append(c10Img)
    else:
        Rest.append(c10Img)


random.shuffle(filteredImages)
random.shuffle(A1Only)
random.shuffle(A2Only)

c10New = filteredImages[0:180] + A1Only[0:10] + A2Only[0:10]

outputFile = os.path.join(splitsDir, "c10new.txt")
with open(outputFile, "w") as f:
    f.writelines(["{}\n".format(x) for x in c10New])

print(len(c10New))

import pdb; pdb.set_trace()

# c10NewImages =
