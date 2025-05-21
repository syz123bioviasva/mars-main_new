import os
import sys
from sklearn.model_selection import train_test_split
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from misc.xml import XmlBbox


trainRatio = 0.8
valRatio = 0.1
testRatio = 0.1

classList = ["A{}".format(x) for x in range(1, 21)]

imageDir = "/auto/cvdata/mar20/images"
annotationDir = "/auto/cvdata/mar20/annotations"
splitDir = "/auto/cvdata/mar20/splits/v2"

imageFiles = [x for x in os.listdir(imageDir) if x.endswith(".jpg")]
annotationFiles = []

for imageFile in imageFiles:
    baseName = os.path.basename(imageFile).split(".")[0]
    annFile = os.path.join(annotationDir, "{}.xml".format(baseName))
    if not os.path.exists(annFile):
        raise ValueError("Annotation file not exists: {}".format(annFile))
    annotationFiles.append(annFile)

excludededClasses = ["A19", "A20"]
excludeded = []

for imgFile, annFile in zip(imageFiles, annotationFiles):
    objList = XmlBbox.loadXmlObjectList(annFile, classList, asArray=False)
    imgClasses = []
    for obj in objList:
        if obj.className in excludededClasses:
            excludeded.append(imgFile)
            break

restImages = [x for x in imageFiles if x not in excludeded]

# split rest images
train, testVal = train_test_split(restImages, test_size=testRatio + valRatio)
test, val = train_test_split(testVal, test_size=valRatio / (testRatio + valRatio))

with open(os.path.join(splitDir, "train.txt"), "w") as f:
    f.writelines(["{}\n".format(x) for x in train])

with open(os.path.join(splitDir, "test.txt"), "w") as f:
    f.writelines(["{}\n".format(x) for x in test])

with open(os.path.join(splitDir, "validation.txt"), "w") as f:
    f.writelines(["{}\n".format(x) for x in val])

# split A19, A20 images
etran, etest = train_test_split(excludeded, test_size=0.2)

with open(os.path.join(splitDir, "e2train.txt"), "w") as f:
    f.writelines(["{}\n".format(x) for x in etran])

with open(os.path.join(splitDir, "e2test.txt"), "w") as f:
    f.writelines(["{}\n".format(x) for x in etest])
