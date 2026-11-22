# Kiến trúc Multi-task Model (Shared Backbone + Gender Head [1 xác suất (0,1)] + Age Group Head [9 nhóm tuổi])
# Hỗ trợ 5 chế độ:
# - 02a (multiclass): Linear(512, 9) -> Softmax
# - 02b (coral):      Linear(512, 8) -> CORAL 8 logits độc lập
# - 02c (coral_v2):   Projection Heads + CORALLayer 1-D đơn điệu
# - 02d (coral_v3):   Multi-Scale Heads (Gender: 512->256->1 | Age: 768->256->8)
# - 02e (coral_v4):   Multi-Scale + Cross-Task Conditioning (Gender 256-d -> Age Head) + Isotonic Monotonicity & Calibrated Thresholds

import torch
import torch.nn as nn
import torch.nn.functional as F


class CORALLayer(nn.Module):
    """Tầng CORAL chuẩn theo Cao et al. (Shared weight w^T h + 8 bias giảm dần)."""

    def __init__(self, in_features, num_classes=9):
        super().__init__()
        self.num_thresholds = num_classes - 1
        self.fc = nn.Linear(in_features, 1, bias=False)
        self.b0 = nn.Parameter(torch.tensor(2.0))
        self.raw_deltas = nn.Parameter(torch.full((self.num_thresholds - 1,), -0.26))

    def get_biases(self):
        gaps = F.softplus(self.raw_deltas)
        decrements = torch.cat([torch.zeros(1, device=gaps.device, dtype=gaps.dtype), torch.cumsum(gaps, dim=0)])
        return self.b0 - decrements

    def forward(self, h):
        return self.fc(h) + self.get_biases().unsqueeze(0)


class MultiTaskAgeGender(nn.Module):
    def __init__(self, backbone, age_mode="coral_v4", num_age_groups=9, dropout=0.3):
        super().__init__()
        assert age_mode in ("multiclass", "coral", "coral_v2", "coral_v3", "coral_v4")
        self.backbone = backbone
        self.age_mode = age_mode
        self.num_age_groups = num_age_groups
        self.has_multiscale = hasattr(backbone, "forward_multiscale")

        if age_mode == "coral_v4":
            self.dropout = nn.Identity()
            g_in = backbone.out_features
            a_in = getattr(backbone, "out_age_features", backbone.out_features)
            self.gender_proj = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(g_in, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.GELU(),
            )
            self.gender_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(256, 1),
            )
            # Cross-Task Conditioning: ghép gender_proj (256-d, detach) vào age_head (768 + 256 = 1024-d)
            self.age_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(a_in + 256, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_age_groups - 1),
            )
        elif age_mode == "coral_v3":
            self.dropout = nn.Identity()
            g_in = backbone.out_features
            a_in = getattr(backbone, "out_age_features", backbone.out_features)
            self.gender_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(g_in, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 1),
            )
            self.age_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(a_in, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_age_groups - 1),
            )
        elif age_mode == "coral_v2":
            self.dropout = nn.Identity()
            self.gender_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(backbone.out_features, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 1),
            )
            self.age_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(backbone.out_features, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(dropout),
                CORALLayer(256, num_classes=num_age_groups),
            )
        else:
            self.dropout = nn.Dropout(dropout)
            self.gender_head = nn.Linear(backbone.out_features, 1)
            age_out_dim = (num_age_groups - 1) if age_mode == "coral" else num_age_groups
            self.age_head = nn.Linear(backbone.out_features, age_out_dim)

    def extract_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        if self.age_mode == "coral_v4":
            if self.has_multiscale:
                feat_g, feat_a = self.backbone.forward_multiscale(x)
            else:
                feat_g = self.extract_features(x)
                feat_a = feat_g
            g_emb = self.gender_proj(feat_g)
            g_logit = self.gender_head(g_emb).squeeze(1)
            a_logits = self.age_head(torch.cat([feat_a, g_emb.detach()], dim=1))
            return g_logit, a_logits

        if self.age_mode == "coral_v3" and self.has_multiscale:
            feat_g, feat_a = self.backbone.forward_multiscale(x)
            return self.gender_head(feat_g).squeeze(1), self.age_head(feat_a)

        feats = self.dropout(self.extract_features(x))
        gender_logit = self.gender_head(feats).squeeze(1)
        age_logits = self.age_head(feats)
        return gender_logit, age_logits

    @torch.no_grad()
    def predict_proba(self, x, tta=False, enforce_monotonic=True, thresholds=None):
        g_logit, a_logits = self.forward(x)
        g_prob = torch.sigmoid(g_logit)
        is_coral = self.age_mode in ("coral", "coral_v2", "coral_v3", "coral_v4")
        a_scores = torch.sigmoid(a_logits) if is_coral else torch.softmax(a_logits, dim=1)

        if tta:
            g_logit_f, a_logits_f = self.forward(torch.flip(x, dims=[3]))
            g_prob = 0.5 * (g_prob + torch.sigmoid(g_logit_f))
            a_scores_f = torch.sigmoid(a_logits_f) if is_coral else torch.softmax(a_logits_f, dim=1)
            a_scores = 0.5 * (a_scores + a_scores_f)

        if is_coral:
            if enforce_monotonic:
                a_scores = torch.cummin(a_scores, dim=1).values
            if thresholds is not None:
                th = torch.as_tensor(thresholds, device=a_scores.device, dtype=a_scores.dtype).unsqueeze(0)
                age_pred_group = (a_scores > th).sum(dim=1).long()
            else:
                age_pred_group = (a_scores > 0.5).sum(dim=1).long()
        else:
            age_pred_group = a_scores.argmax(dim=1).long()
        return g_prob, age_pred_group, a_scores
