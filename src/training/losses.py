# Hàm mất mát Multi-task (Multi-task Loss) theo hướng Báo cáo PDF:
# - Nhánh Giới tính: BCEWithLogitsLoss + Label Smoothing (alpha = 0.1) cho đầu ra 1 xác suất (0, 1)
# - Nhánh Tuổi (9 nhóm tuổi 0..8):
#   + Kiểu A (age_mode="multiclass"): CrossEntropyLoss (9 lớp)
#   + Kiểu B (age_mode="coral"): CORAL Loss (8 bài toán nhị phân liên tiếp P(y > k))
# - Total Loss = w_gender * loss_gender + w_age * loss_age

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    def __init__(self, age_mode="coral", num_age_groups=9, w_gender=1.0, w_age=3.0, label_smoothing=0.1):
        super().__init__()
        assert age_mode in ("multiclass", "coral"), "age_mode phải là 'multiclass' hoặc 'coral'"
        self.age_mode = age_mode
        self.num_age_groups = num_age_groups
        self.w_gender = w_gender
        self.w_age = w_age
        self.label_smoothing = label_smoothing
        self.bce_gender = nn.BCEWithLogitsLoss()
        self.ce_age = nn.CrossEntropyLoss()

    def gender_loss(self, gender_logit, gender_true):
        g_target = gender_true.float()
        if self.label_smoothing > 0:
            g_target = g_target * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        return self.bce_gender(gender_logit, g_target)

    def age_loss(self, age_logits, age_group_true):
        if self.age_mode == "coral":
            # Tạo ma trận nhị phân (B, K-1=8): cột k = 1 nếu age_group_true > k
            thresholds = torch.arange(self.num_age_groups - 1, device=age_logits.device).unsqueeze(0)
            levels = (age_group_true.unsqueeze(1) > thresholds).float()
            return F.binary_cross_entropy_with_logits(age_logits, levels)
        return self.ce_age(age_logits, age_group_true.long())

    def forward(self, gender_logit, age_logits, gender_true, age_group_true, return_parts=False):
        lg = self.gender_loss(gender_logit, gender_true)
        la = self.age_loss(age_logits, age_group_true)
        total = self.w_gender * lg + self.w_age * la
        if return_parts:
            return total, lg, la
        return total
