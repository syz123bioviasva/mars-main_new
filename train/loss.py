import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.bbox import bboxDecode, iou, bbox2dist
from train.tal import TaskAlignedAssigner


class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iouv = iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iouv) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class DetectionLoss(object):
    def __init__(self, mcfg, model):
        self.model = model
        self.mcfg= mcfg
        self.layerStrides = model.layerStrides
        self.assigner = TaskAlignedAssigner(topk=self.mcfg.talTopk, num_classes=self.mcfg.nc, alpha=0.5, beta=6.0)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.bboxLoss = BboxLoss(self.mcfg.regMax).to(self.mcfg.device)

    def preprocess(self, targets, batchSize, scaleTensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batchSize, 0, ne - 1, device=self.mcfg.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batchSize, counts.max(), ne - 1, device=self.mcfg.device)
            for j in range(batchSize):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = out[..., 1:5].mul_(scaleTensor)
        return out

    def __call__(self, preds, targets):
        """
        preds shape:
            preds[0]: (B, regMax * 4 + nc, 80, 80)
            preds[1]: (B, regMax * 4 + nc, 40, 40)
            preds[2]: (B, regMax * 4 + nc, 20, 20)
        targets shape:
            (?, 6)
        """
        loss = torch.zeros(3, device=self.mcfg.device)  # box, cls, dfl

        batchSize = preds[0].shape[0]
        no = self.mcfg.nc + self.mcfg.regMax * 4
        
        # predictioin preprocess
        predBoxDistribution, predClassScores = torch.cat([xi.view(batchSize, no, -1) for xi in preds], 2).split((self.mcfg.regMax * 4, self.mcfg.nc), 1)
        predBoxDistribution = predBoxDistribution.permute(0, 2, 1).contiguous() # (batchSize, 80 * 80 + 40 * 40 + 20 * 20, regMax * 4)
        predClassScores = predClassScores.permute(0, 2, 1).contiguous() # (batchSize, 80 * 80 + 40 * 40 + 20 * 20, nc)
        anchor_points, stride_tensor = self.model.anchorPoints, self.model.anchorStrides
        # ground truth preprocess
        dtype = predClassScores.dtype
        targets = self.preprocess(targets.to(self.mcfg.device), batchSize, scaleTensor=self.model.scaleTensor) # (batchSize, maxCount, 5)
        gtLabels, gtBboxes = targets.split((1, 4), 2)  # cls=(batchSize, maxCount, 1), xyxy=(batchSize, maxCount, 4)
        gtMask = gtBboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = bboxDecode(anchor_points, predBoxDistribution, self.model.proj)  # xyxy, (b, h*w, 4)
 
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            predClassScores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gtBboxes.dtype),
            anchor_points * stride_tensor,
            gtLabels,
            gtBboxes,
            gtMask,
        )
        target_scores_sum = max(target_scores.sum(), 1)
 
        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(predClassScores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
 
        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bboxLoss(
                predBoxDistribution, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        #raise NotImplementedError("DetectionLoss::__call__")

        loss[0] *= self.mcfg.lossWeights[0]  # box
        loss[1] *= self.mcfg.lossWeights[1]  # cls
        loss[2] *= self.mcfg.lossWeights[2]  # dfl

        return loss.sum()
