# Hàm mất mát Multi-task (Multi-task Loss):
# - Nhánh Giới tính: BCEWithLogitsLoss + Label Smoothing (alpha = 0.1) cho đầu ra 1 xác suất (0, 1)
# - Nhánh Tuổi (9 nhóm tuổi 0..8):
#   + Kiểu A   (multiclass): CrossEntropyLoss (9 lớp)
#   + Kiểu B   (coral):      CORAL BCEWithLogitsLoss (8 ngưỡng)
#   + Kiểu B+  (coral_v2):   Focal CORAL Loss (gamma=1.5)
#   + Kiểu B++ (coral_v3):   Weighted BCE chuẩn (tắt Focal để không bẹt xác suất 0-2) + Ràng buộc thứ tự mềm (Monotonic Penalty)

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        age_mode="coral_v3",
        num_age_groups=9,
        w_gender=1.0,
        w_age=3.0,
        label_smoothing=0.1,
        focal_gamma=0.0,
        mono_lambda=0.25,
        threshold_weights=None,
    ):
        super().__init__()
        assert age_mode in ("multiclass", "coral", "coral_v2", "coral_v3")
        self.age_mode = age_mode
        self.num_age_groups = num_age_groups
        self.w_gender = w_gender
        self.w_age = w_age
        self.label_smoothing = label_smoothing
        self.focal_gamma = 1.5 if (age_mode == "coral_v2" and focal_gamma == 0.0) else focal_gamma
        self.mono_lambda = mono_lambda
        self.bce_gender = nn.BCEWithLogitsLoss()
        self.ce_age = nn.CrossEntropyLoss()

        if threshold_weights is None:
            threshold_weights = [1.0, 1.0, 1.4, 1.4, 1.15, 1.0, 1.0, 1.0]
        self.register_buffer("threshold_weights", torch.tensor(threshold_weights, dtype=torch.float32))

    def gender_loss(self, gender_logit, gender_true):
        g_target = gender_true.float()
        if self.label_smoothing > 0:
            g_target = g_target * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        return self.bce_gender(gender_logit, g_target)

    def age_loss(self, age_logits, age_group_true):
        if self.age_mode == "coral":
            thresholds = torch.arange(self.num_age_groups - 1, device=age_logits.device).unsqueeze(0)
            levels = (age_group_true.unsqueeze(1) > thresholds).float()
            return F.binary_cross_entropy_with_logits(age_logits, levels)

        if self.age_mode == "coral_v2":
            thresholds = torch.arange(self.num_age_groups - 1, device=age_logits.device).unsqueeze(0)
            levels = (age_group_true.unsqueeze(1) > thresholds).float()
            bce_mat = F.binary_cross_entropy_with_logits(age_logits, levels, reduction="none")
            pt = torch.exp(-bce_mat)
            focal_mod = (1.0 - pt) ** self.focal_gamma
            w = self.threshold_weights.to(age_logits.device).unsqueeze(0)
            return (bce_mat * focal_mod * w).sum() / (w.sum() * age_logits.size(0))

        if self.age_mode == "coral_v3":
            thresholds = torch.arange(self.num_age_groups - 1, device=age_logits.device).unsqueeze(0)
            levels = (age_group_true.unsqueeze(1) > thresholds).float()
            bce_mat = F.binary_cross_entropy_with_logits(age_logits, levels, reduction="none")
            w = self.threshold_weights.to(age_logits.device).unsqueeze(0)
            base_loss = (bce_mat * w).sum() / (w.sum() * age_logits.size(0))
            # Ràng buộc thứ tự mềm: phạt nhẹ nếu p_{k+1} > p_k
            probs = torch.sigmoid(age_logits)
            mono_penalty = torch.relu(probs[:, 1:] - probs[:, :-1]).mean()
            return base_loss + self.mono_lambda * mono_penalty

        return self.ce_age(age_logits, age_group_true.long())

    def forward(self, gender_logit, age_logits, gender_true, age_group_true, return_parts=False):
        lg = self.gender_loss(gender_logit, gender_true)
        la = self.age_loss(age_logits, age_group_true)
        total = self.w_gender * lg + self.w_age * la
        if return_parts:
            return total, lg, la
        return total
