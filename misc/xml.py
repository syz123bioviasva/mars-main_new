import numpy as np
import xml.etree.ElementTree as ET


class XmlBbox(object):
    def __init__(self, className, classIndex, xmin, ymin, xmax, ymax):
        self.className = className
        self.classIndex = classIndex
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def asArray(self):
        return np.array([self.xmin, self.ymin, self.xmax, self.ymax, self.classIndex], dtype=np.integer)

    @classmethod
    def loadFromXmlItem(cls, item, classList, selectedClasses=None):
        nameObj = item.find("name")
        bboxObj = item.find("bndbox")
        if nameObj is None or bboxObj is None or nameObj.text not in classList:
            return None
        if selectedClasses is not None and nameObj.text not in selectedClasses:
            return None
        xminObj = bboxObj.find("xmin")
        xmaxObj = bboxObj.find("xmax")
        yminObj = bboxObj.find("ymin")
        ymaxObj = bboxObj.find("ymax")
        if xminObj is None or xmaxObj is None or yminObj is None or ymaxObj is None:
            return None
        return XmlBbox(
            className=nameObj.text,
            classIndex=classList.index(nameObj.text),
            xmin=int(float(xminObj.text)),
            ymin=int(float(yminObj.text)),
            xmax=int(float(xmaxObj.text)),
            ymax=int(float(ymaxObj.text)),
        )

    @classmethod
    def loadXmlObjectList(cls, xmlFile, classList, selectedClasses=None, asArray=False):
        root = ET.parse(xmlFile).getroot()
        if root is None:
            raise ValueError("Empty xml file: {}".format(xmlFile))
        retList = []
        for item in root.iter("object"):
            xmlObj = cls.loadFromXmlItem(item, classList, selectedClasses)
            if xmlObj is not None:
                retList.append(xmlObj)
        if asArray:
            return np.array([x.asArray() for x in retList])
        return retList
