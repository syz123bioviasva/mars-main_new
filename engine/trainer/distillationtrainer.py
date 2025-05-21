import torch
from engine.trainer.base import BaseTrainer
from train.distilloss import DistillationLoss

class DistillationTrainer(BaseTrainer):
    def __init__(self, config, model):
        super(DistillationTrainer, self).__init__(config, model)
        self.distill_loss = DistillationLoss(config)
        
    def train_step(self, batch):
        """
        执行一步训练
        Args:
            batch: 包含图像和标签的批次数据
        """
        images, targets = batch
        
        # 将数据移到设备
        images = images.to(self.device)
        targets = [target.to(self.device) for target in targets]
        
        # 前向传播
        with torch.no_grad():
            teacher_outputs = self.model.teacherModel(images)
        
        student_outputs = self.model(images)
        
        # 计算蒸馏损失
        loss, loss_dict = self.distill_loss(
            student_outputs,
            teacher_outputs,
            targets
        )
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), loss_dict
    
    def validate_step(self, batch):
        """
        执行一步验证
        Args:
            batch: 包含图像和标签的批次数据
        """
        images, targets = batch
        
        # 将数据移到设备
        images = images.to(self.device)
        targets = [target.to(self.device) for target in targets]
        
        # 前向传播
        with torch.no_grad():
            student_outputs = self.model(images)
            teacher_outputs = self.model.teacherModel(images)
            
            # 计算蒸馏损失
            loss, loss_dict = self.distill_loss(
                student_outputs,
                teacher_outputs,
                targets
            )
        
        return loss.item(), loss_dict 