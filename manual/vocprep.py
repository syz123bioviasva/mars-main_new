import os
from sklearn.model_selection import train_test_split


trainRatio = 0.8
valRatio = 0.1
testRatio = 0.1

imageDir = "/auto/cvdata/mar20/images"
splitDir = "/auto/cvdata/mar20/splits/v1"

imageFiles = [x for x in os.listdir(imageDir) if x.endswith(".jpg")]

train, testVal = train_test_split(imageFiles, test_size=testRatio + valRatio)
test, val = train_test_split(testVal, test_size=valRatio / (testRatio + valRatio))

trainFile = os.path.join(splitDir, "train.txt")
testFile = os.path.join(splitDir, "test.txt")
valFile = os.path.join(splitDir, "validation.txt")

with open(trainFile, "w") as f:
    f.writelines(["{}\n".format(x) for x in train])

with open(testFile, "w") as f:
    f.writelines(["{}\n".format(x) for x in test])

with open(valFile, "w") as f:
    f.writelines(["{}\n".format(x) for x in val])
