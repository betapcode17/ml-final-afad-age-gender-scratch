# Kiến trúc Multi-task Model (Shared Backbone + Gender Head [1 xác suất (0,1)] + Age Group Head [9 nhóm tuổi])
# Hỗ trợ 2 kiểu nhánh Tuổi để so sánh:
# - Kiểu A (age_mode="multiclass"): Linear(512, 9) -> Softmax dự đoán 1 trong 9 nhóm tuổi
# - Kiểu B (age_mode="coral"): Linear(512, 8) -> CORAL Ordinal Regression (8 ngưỡng nhị phân) suy ra 1 trong 9 nhóm tuổi

import torch
import torch.nn as nn


class MultiTaskAgeGender(nn.Module):
    def __init__(self, backbone, age_mode="coral", num_age_groups=9, dropout=0.3):
        super().__init__()
        assert age_mode in ("multiclass", "coral"), "age_mode phải là 'multiclass' hoặc 'coral'"
        self.backbone = backbone
        self.age_mode = age_mode
        self.num_age_groups = num_age_groups
        self.dropout = nn.Dropout(dropout)

        # Nhánh Giới tính: 1 logit duy nhất -> qua Sigmoid trả về 1 xác suất P(Male) trong (0, 1)
        self.gender_head = nn.Linear(backbone.out_features, 1)

        # Nhánh Tuổi: 9 logits (Kiểu A - Multi-class) hoặc 8 logits (Kiểu B - CORAL Ordinal)
        age_out_dim = (num_age_groups - 1) if age_mode == "coral" else num_age_groups
        self.age_head = nn.Linear(backbone.out_features, age_out_dim)

    def extract_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        feats = self.dropout(self.extract_features(x))
        gender_logit = self.gender_head(feats).squeeze(1)  # (B,)
        age_logits = self.age_head(feats)                  # (B, 9) hoặc (B, 8)
        return gender_logit, age_logits

    @torch.no_grad()
    def predict_proba(self, x):
        """Trả về:
        - gender_prob: Tensor (B,) chứa 1 xác suất trong khoảng (0, 1) [P(Male=1)]
        - age_pred_group: Tensor (B,) chứa chỉ số nhóm tuổi dự đoán (0..8)
        - age_scores: Tensor (B, 9) xác suất Softmax hoặc (B, 8) xác suất ngưỡng CORAL
        """
        gender_logit, age_logits = self.forward(x)
        gender_prob = torch.sigmoid(gender_logit)
        if self.age_mode == "coral":
            age_scores = torch.sigmoid(age_logits)
            age_pred_group = (age_scores > 0.5).sum(dim=1).long()
        else:
            age_scores = torch.softmax(age_logits, dim=1)
            age_pred_group = age_scores.argmax(dim=1).long()
        return gender_prob, age_pred_group, age_scores
