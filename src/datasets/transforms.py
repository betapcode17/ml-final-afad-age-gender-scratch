# Data Augmentation — Làm giàu dữ liệu cho tập Train và chuẩn hóa cho Val/Test
# Yêu cầu:
# - Tập Train: RandomHorizontalFlip, RandomRotation (quay góc), ColorJitter (độ sáng/tương phản),
#   RandomVerticalFlip (tùy chọn thử nghiệm theo ghi chú), Resize, ToTensor, Normalize
# - Tập Val/Test: Chỉ Resize, ToTensor, Normalize (không augment)
