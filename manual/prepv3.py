import os
import random
from sklearn.model_selection import train_test_split


d1 = "/auto/cvdata/mar20/splits/misc/2"
d2 = "/auto/cvdata/mar20/splits/misc/18"

imageSet = set()

for dpath in (d1, d2):
    for subfile in ("train.txt", "test.txt", "validation.txt"):
        file = os.path.join(dpath, subfile)
        lines = open(file).readlines()
        lines = [x.strip() for x in lines]
        lines = [x for x in lines if len(x) > 0]
        for line in lines:
            imageSet.add(line)


with open("/auto/cvdata/mar20/splits/v3/c10.txt", "w") as f:
    for img in imageSet:
        f.write("{}\n".format(img))


imageDir = "/auto/cvdata/mar20/images"
allImages = os.listdir(imageDir)
random.shuffle(allImages)

testImages = set()
restImages = set()

for image in allImages:
    if image not in imageSet:
        testImages.add(image)
        if len(testImages) >= 400:
            break

for image in allImages:
    if image not in testImages:
        restImages.add(image)

# train, testVal = train_test_split(imageFiles, test_size=testRatio + valRatio)
train, val = train_test_split(list(restImages), test_size=0.11)

with open("/auto/cvdata/mar20/splits/v3/train.txt", "w") as f:
    for img in train:
        f.write("{}\n".format(img))

with open("/auto/cvdata/mar20/splits/v3/validation.txt", "w") as f:
    for img in val:
        f.write("{}\n".format(img))

with open("/auto/cvdata/mar20/splits/v3/test.txt", "w") as f:
    for img in testImages:
        f.write("{}\n".format(img))


import pdb; pdb.set_trace()
