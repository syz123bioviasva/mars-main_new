import math
import torch
import torchvision
import numpy as np


def rescaleBoxes(boxList, oriWidth, oriHeight, scaledWidth, scaledHeight, targetWidth, targetHeight, xoffset, yoffset, flip):
    if len(boxList) > 0:
        np.random.shuffle(boxList)
        boxList[:, [0, 2]] = boxList[:, [0, 2]] * scaledWidth / oriWidth + xoffset
        boxList[:, [1, 3]] = boxList[:, [1, 3]] * scaledHeight / oriHeight + yoffset
        if flip:
            boxList[:, [0,2]] = targetWidth - boxList[:, [2,0]]
        boxList[:, 0 : 2][boxList[:, 0 : 2] < 0] = 0
        boxList[:, 2][boxList[:, 2] > targetWidth] = targetWidth
        boxList[:, 3][boxList[:, 3] > targetHeight] = targetHeight
        boxWidthList = boxList[:, 2] - boxList[:, 0]
        boxHeightList = boxList[:, 3] - boxList[:, 1]
        boxList = boxList[np.logical_and(boxWidthList > 1, boxHeightList > 1)] # discard invalid boxList
    return boxList


def recoverBoxes(boxList, oriWidth, oriHeight, scaledWidth, scaledHeight, targetWidth, targetHeight, xoffset, yoffset, flip):
    """
    Oppsite opeartion of "rescaleBoxes(...)"
    """
    if len(boxList) > 0:
        boxList[:, [0, 2]] = (boxList[:, [0, 2]] - xoffset) * oriWidth / scaledWidth
        boxList[:, [1, 3]] = (boxList[:, [1, 3]] - yoffset) * oriHeight / scaledHeight
        if flip:
            boxList[:, [0,2]] = oriWidth - boxList[:, [2,0]]
        boxList[:, 0 : 2][boxList[:, 0 : 2] < 0] = 0
        boxList[:, 2][boxList[:, 2] > oriWidth] = oriWidth
        boxList[:, 3][boxList[:, 3] > oriHeight] = oriHeight
    return boxList


def isValidBox(box):
    if any(box < 0):
        return False
    if box[2] <= box[0]:
        return False
    if box[3] < box[1]:
        return False
    return True


def makeAnchors(featShapes, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    for i, stride in enumerate(strides):
        h, w = featShapes[i]
        sx = torch.arange(end=w) + grid_cell_offset  # shift x
        sy = torch.arange(end=h) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh is True:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)  # dist (lt, rb)

'''
def bboxDecode(anchor_points, pred_dist, proj, xywh):
    """Decode predicted object bounding box coordinates from anchor points and distribution."""
    b, a, c = pred_dist.shape  # batch, anchors, channels
    pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(proj.type(pred_dist.dtype))
    return dist2bbox(pred_dist, anchor_points, xywh=xywh)
'''
def bboxDecode(anchor_points, pred_dist, proj):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        b, a, c = pred_dist.shape  # batch, anchors, channels
        pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)


def iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).

    Args:
        box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
        box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format. Defaults to True.
        GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
        DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
        CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
            ) / 4  # center dist**2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


def nonMaxSuppression(
    predClassScores,
    predBboxes,
    scoreThres,
    iouThres,
    maxDetect,
):
    """
    Parameters shape:
        predClassScores: (batchSize, 8400, nc)
        predBboxes: (batchSize, 8400, 4)
    """
    batchSize = predClassScores.shape[0]
    device = predClassScores.device
    scores, predClasses = torch.max(predClassScores, 2, keepdim=True)
    scoreMask = (scores > scoreThres).squeeze()
    outputList = [torch.zeros(0, 6).to(device)] * batchSize

    for i in range(batchSize):
        imgScoreMask = scoreMask[i]
        if not imgScoreMask.any():
            continue
        imgClasses = predClasses[i]
        imgScores = scores[i]
        imgBboxes = predBboxes[i]
        filteredClasses = imgClasses[imgScoreMask]
        filteredBboxes = imgBboxes[imgScoreMask]
        filteredScores = imgScores[imgScoreMask]
        nmsIndex = torchvision.ops.nms(filteredBboxes, filteredScores.squeeze(dim=1), iouThres)
        nmsIndex = nmsIndex[:maxDetect]  # limit detections
        result = torch.cat([filteredClasses[nmsIndex], filteredScores[nmsIndex], filteredBboxes[nmsIndex]], axis=1).to(device) # (N, 6)
        outputList[i] = result

    return outputList
