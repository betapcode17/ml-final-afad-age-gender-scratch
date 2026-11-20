# Hàm mất mát Multi-task (Multi-task Loss)
# Yêu cầu:
# - Gender Loss (Classification): CrossEntropyLoss
# - Age Loss (Regression): SmoothL1Loss (Huber) hoặc L1Loss (MAE)
# - Total Loss = w_gender * loss_gender + w_age * loss_age
