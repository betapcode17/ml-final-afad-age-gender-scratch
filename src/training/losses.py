# Hàm mất mát Multi-task (Multi-task Loss)
# Yêu cầu:
# - Gender Loss (Classification): CrossEntropyLoss
# - Age Loss (Regression): SmoothL1Loss (Huber) hoặc L1Loss (MAE)
# - Total Loss = w_gender * loss_gender + w_age * loss_age


import torch.nn as nn
class MultiTaskLoss(nn.Module):
    def __init__(self, w_gender=1.0, w_age=0.1):
        super().__init__()
        self.w_gender = w_gender
        self.w_age = w_age
        self.ce = nn.CrossEntropyLoss()
        self.huber = nn.SmoothL1Loss()
    def forward(self, gender_logits, age_pred, gender_true, age_true):
        return self.w_gender * self.ce(gender_logits, gender_true) + self.w_age * self.huber(age_pred,age_true)