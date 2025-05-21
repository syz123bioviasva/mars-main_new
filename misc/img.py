import cv2
import numpy as np
from PIL import Image


def loadRGBImage(imFile):
    image = Image.open(imFile)
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert("RGB")
        return image


def rescale2Target(image, scaledWidth, scaledHeight, targetWidth, targetHeight):
    xoffset = int((targetWidth - scaledWidth) / 2)
    yoffset = int((targetHeight - scaledHeight) / 2)
    newImage = Image.new("RGB", (targetWidth, targetHeight), (128, 128, 128))
    scaledImage = image.resize((scaledWidth, scaledHeight), Image.BICUBIC)
    newImage.paste(scaledImage, (xoffset, yoffset))
    return newImage, xoffset, yoffset


def hsvAdjust(image, huef, satf, valf):
    imageData = np.array(image, np.uint8)
    r = np.random.uniform(-1, 1, 3) * [huef, satf, valf] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(imageData, cv2.COLOR_RGB2HSV))
    x = np.arange(0, 256, dtype=r.dtype)
    lutHue = ((x * r[0]) % 180).astype(imageData.dtype)
    lutSat = np.clip(x * r[1], 0, 255).astype(imageData.dtype)
    lutVal = np.clip(x * r[2], 0, 255).astype(imageData.dtype)
    imageData = cv2.merge((cv2.LUT(hue, lutHue), cv2.LUT(sat, lutSat), cv2.LUT(val, lutVal)))
    imageData = cv2.cvtColor(imageData, cv2.COLOR_HSV2RGB)
    return imageData
