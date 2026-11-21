# Kiến trúc Multi-task Model (Shared Backbone + Age Head + Gender Head)
# Yêu cầu:
# - Nhận backbone (ResNet-18 hoặc MobileNetV2)
# - Nhánh 1 - Gender Head (Classification): Linear -> 2 logits (qua Softmax khi suy luận để ra CHUỖI XÁC XUẤT)
# - Nhánh 2 - Age Head (Regression): Linear -> 1 giá trị tuổi liên tục
# - Hàm forward(x) trả về (gender_logits, age_pred)

import torch 
import torch.nn as nn

class MultiTaskAgeGender(nn.Module):
    def __init__(self, backbone, num_genders=2, dropout = 0.3):
        super().__init__()
        self.backbone = backbone 
        self.dropout = nn.Dropout(dropout)
        self.gender_head = nn.Linear(backbone.out_features,num_genders)
        self.age_head = nn.Linear(backbone.out_features,1)
    
    def forward(self, x):
        feats = self.dropout(self.backbone(x))
        gender_logits = self.gender_head(feats)
        age_pred = self.age_head(feats).squeeze(1)
        return gender_logits, age_pred

    @torch.no_grad()
    def predict_proba(self, x):

        gender_logits, age_pred = self.forward(x)
        gender_probs = torch.softmax(gender_logits, dim=1)
        return gender_probs, age_pred   