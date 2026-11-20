# Kiến trúc Multi-task Model (Shared Backbone + Age Head + Gender Head)
# Yêu cầu:
# - Nhận backbone (ResNet-18 hoặc MobileNetV2)
# - Nhánh 1 - Gender Head (Classification): Linear -> 2 logits (qua Softmax khi suy luận để ra CHUỖI XÁC XUẤT)
# - Nhánh 2 - Age Head (Regression): Linear -> 1 giá trị tuổi liên tục
# - Hàm forward(x) trả về (gender_logits, age_pred)
