# Hàm mất mát Multi-task (Multi-task Loss):
# - 02a (multiclass): CrossEntropy 9 lớp
# - 02b (coral):      CORAL BCEWithLogitsLoss (8 ngưỡng)
# - 02c (coral_v2):   Focal CORAL Loss (gamma=1.5)
# - 02d (coral_v3):   Weighted BCE + Phạt thứ tự mềm (Monotonic Penalty)
# - 02e (coral_v4):   Soft Ordinal Labels (LDL theo tuổi thực) + Weighted BCE + Mono Penalty + Kendall Uncertainty Weighting

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.datasets.afad_dataset import AGE_CUTOFFS_8


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        age_mode="coral_v4",
        num_age_groups=9,
        w_gender=1.0,
        w_age=3.0,
        label_smoothing=0.1,
        focal_gamma=0.0,
        mono_lambda=0.25,
        ldl_tau=1.5,
        fairface_smooth=0.03,
        threshold_weights=None,
    ):
        super().__init__()
        assert age_mode in ("multiclass", "coral", "coral_v2", "coral_v3", "coral_v4")
        self.age_mode = age_mode
        self.num_age_groups = num_age_groups
        self.w_gender = w_gender
        self.w_age = w_age
        self.label_smoothing = label_smoothing
        self.focal_gamma = 1.5 if (age_mode == "coral_v2" and focal_gamma == 0.0) else focal_gamma
        self.mono_lambda = mono_lambda
        self.ldl_tau = ldl_tau
        self.fairface_smooth = fairface_smooth
        self.bce_gender = nn.BCEWithLogitsLoss()
        self.ce_age = nn.CrossEntropyLoss()

        if threshold_weights is None:
            threshold_weights = [1.0, 1.0, 1.4, 1.4, 1.15, 1.0, 1.0, 1.0]
        self.register_buffer("threshold_weights", torch.tensor(threshold_weights, dtype=torch.float32))
        self.register_buffer("age_cutoffs", torch.tensor(AGE_CUTOFFS_8, dtype=torch.float32))

        # Kendall Uncertainty Weighting parameters cho coral_v4 (clamp [-4, 4] để ổn định)
        self.log_var_gender = nn.Parameter(torch.tensor(0.0))
        self.log_var_age = nn.Parameter(torch.tensor(-0.5))

    def gender_loss(self, gender_logit, gender_true):
        g_target = gender_true.float()
        if self.label_smoothing > 0:
            g_target = g_target * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        return self.bce_gender(gender_logit, g_target)

    def build_soft_ordinal_targets(self, age_group_true, age_exact=None, has_exact=None):
        thresholds = torch.arange(self.num_age_groups - 1, device=age_group_true.device).unsqueeze(0)
        hard_levels = (age_group_true.unsqueeze(1) > thresholds).float()
        if age_exact is None or has_exact is None:
            return hard_levels * (1.0 - 2.0 * self.fairface_smooth) + self.fairface_smooth
        cutoffs = self.age_cutoffs.to(age_group_true.device).unsqueeze(0)
        soft_exact = torch.sigmoid((age_exact.float().unsqueeze(1) - cutoffs) / self.ldl_tau)
        hard_smooth = hard_levels * (1.0 - 2.0 * self.fairface_smooth) + self.fairface_smooth
        mask = has_exact.float().unsqueeze(1)
        return mask * soft_exact + (1.0 - mask) * hard_smooth

    def age_loss(self, age_logits, age_group_true, age_exact=None, has_exact=None):
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
            probs = torch.sigmoid(age_logits)
            mono_penalty = torch.relu(probs[:, 1:] - probs[:, :-1]).mean()
            return base_loss + self.mono_lambda * mono_penalty

        if self.age_mode == "coral_v4":
            targets = self.build_soft_ordinal_targets(age_group_true, age_exact, has_exact)
            bce_mat = F.binary_cross_entropy_with_logits(age_logits, targets, reduction="none")
            w = self.threshold_weights.to(age_logits.device).unsqueeze(0)
            base_loss = (bce_mat * w).sum() / (w.sum() * age_logits.size(0))
            probs = torch.sigmoid(age_logits)
            mono_penalty = torch.relu(probs[:, 1:] - probs[:, :-1]).mean()
            return base_loss + self.mono_lambda * mono_penalty

        return self.ce_age(age_logits, age_group_true.long())

    def forward(self, gender_logit, age_logits, gender_true, age_group_true, age_exact=None, has_exact=None, return_parts=False):
        lg = self.gender_loss(gender_logit, gender_true)
        la = self.age_loss(age_logits, age_group_true, age_exact=age_exact, has_exact=has_exact)
        if self.age_mode == "coral_v4":
            sg = torch.clamp(self.log_var_gender, -4.0, 4.0)
            sa = torch.clamp(self.log_var_age, -4.0, 4.0)
            total = 0.5 * torch.exp(-sg) * lg + 0.5 * sg + 0.5 * torch.exp(-sa) * la + 0.5 * sa
        else:
            total = self.w_gender * lg + self.w_age * la
        if return_parts:
            return total, lg, la
        return total
