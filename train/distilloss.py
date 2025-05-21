import torch
import torch.nn as nn
import torch.nn.functional as F
from overrides import override # this could be removed since Python 3.12


class DistillationLoss(nn.Module):
    def __init__(self, config):
        super(DistillationLoss, self).__init__()
        self.config = config
        
        # 蒸馏损失权重
        self.feature_weight = config.distilLossWeights[0]  # 特征蒸馏权重
        self.output_weight = config.distilLossWeights[1]   # 输出蒸馏权重
        self.task_weight = config.distilLossWeights[2]     # 任务损失权重
        
        # 温度参数
        self.temperature = 2.0
        
    def forward(self, student_outputs, teacher_outputs, targets):
        """
        计算蒸馏损失
        Args:
            student_outputs: 学生模型输出
            teacher_outputs: 教师模型输出
            targets: 真实标签
        """
        # 1. 特征蒸馏损失
        feature_loss = self._compute_feature_loss(
            student_outputs['features'],
            teacher_outputs['features']
        )
        
        # 2. 输出蒸馏损失
        output_loss = self._compute_output_loss(
            student_outputs['outputs'],
            teacher_outputs['outputs']
        )
        
        # 3. 任务特定损失（检测损失）
        task_loss = self._compute_task_loss(
            student_outputs['outputs'],
            targets
        )
        
        # 总损失
        total_loss = (
            self.feature_weight * feature_loss +
            self.output_weight * output_loss +
            self.task_weight * task_loss
        )
        
        return total_loss, {
            'feature_loss': feature_loss.item(),
            'output_loss': output_loss.item(),
            'task_loss': task_loss.item()
        }
    
    def _compute_feature_loss(self, student_features, teacher_features):
        """计算特征蒸馏损失"""
        loss = 0
        for s_feat, t_feat in zip(student_features, teacher_features):
            # 使用MSE损失
            loss += F.mse_loss(s_feat, t_feat)
        return loss / len(student_features)
    
    def _compute_output_loss(self, student_outputs, teacher_outputs):
        """计算输出蒸馏损失"""
        # 使用KL散度损失
        student_logits = student_outputs / self.temperature
        teacher_logits = teacher_outputs / self.temperature
        
        loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        return loss
    
    def _compute_task_loss(self, outputs, targets):
        """计算任务特定损失（检测损失）"""
        # 这里使用原有的检测损失计算方式
        # 可以根据需要调整损失计算方式
        return self._compute_detection_loss(outputs, targets)
    
    def _compute_detection_loss(self, outputs, targets):
        """计算检测损失"""
        # 实现检测损失计算
        # 包括分类损失、边界框损失等
        pass

class CWDLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, sfeats, tfeats):
        """
        计算特征蒸馏损失
        Args:
            sfeats: 学生模型特征
            tfeats: 教师模型特征
        """
        loss = 0
        for s_feat, t_feat in zip(sfeats, tfeats):
            # 使用MSE损失
            loss += F.mse_loss(s_feat, t_feat)
        return loss / len(sfeats)


class ResponseLoss(nn.Module):
    def __init__(self, device, num_classes, teacher_class_indexes):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.teacher_class_indexes = teacher_class_indexes
        self.temperature = 2.0

    def forward(self, sresponse, tresponse):
        """
        计算响应蒸馏损失
        Args:
            sresponse: 学生模型输出
            tresponse: 教师模型输出
        """
        loss = 0
        for s_resp, t_resp in zip(sresponse, tresponse):
            # 只对教师模型对应的类别计算KL散度
            s_logits = s_resp[:, self.teacher_class_indexes] / self.temperature
            t_logits = t_resp[:, self.teacher_class_indexes] / self.temperature
            
            loss += F.kl_div(
                F.log_softmax(s_logits, dim=-1),
                F.softmax(t_logits, dim=-1),
                reduction='batchmean'
            ) * (self.temperature ** 2)
        
        return loss / len(sresponse)


class DistillationDetectionLoss(object):
    def __init__(self, mcfg, model):
        self.mcfg = mcfg
        self.histMode = False
        from train.loss import DetectionLoss
        self.detectionLoss = DetectionLoss(mcfg, model)
        self.cwdLoss = CWDLoss(self.mcfg.device)
        self.respLoss = ResponseLoss(self.mcfg.device, self.mcfg.nc, self.mcfg.teacherClassIndexes)

    @override
    def __call__(self, rawPreds, batch):
        """
        rawPreds[0] & rawPreds[1] shape: (
            (B, regMax * 4 + nc, 80, 80),
            (B, regMax * 4 + nc, 40, 40),
            (B, regMax * 4 + nc, 20, 20),
            (B, 128 * w, 160, 160),
            (B, 256 * w, 80, 80),
            (B, 512 * w, 40, 40),
            (B, 512 * w * r, 20, 20),
            (B, 512 * w, 40, 40),
            (B, 256 * w, 80, 80),
            (B, 512 * w, 40, 40),
            (B, 512 * w * r, 20, 20),
        )
        """
        spreds = rawPreds[0]
        tpreds = rawPreds[1]

        sresponse, sfeats = spreds[:3], spreds[3:]
        tresponse, tfeats = tpreds[:3], tpreds[3:]

        loss = torch.zeros(3, device=self.mcfg.device)  # original, cwd distillation, response distillation
        loss[0] = self.detectionLoss(sresponse, batch) * self.mcfg.distilLossWeights[0]  # original
        loss[1] = self.cwdLoss(sfeats, tfeats) * self.mcfg.distilLossWeights[1]  # cwd distillation
        loss[2] = self.respLoss(sresponse, tresponse) * self.mcfg.distilLossWeights[2]  # response distillation

        return loss.sum()
