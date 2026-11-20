# Kiến trúc ResNet-18 Backbone (Train từ đầu - KHÔNG dùng pretrained)
# Yêu cầu:
# - Tự định nghĩa BasicBlock (Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN + Skip Connection)
# - Tự định nghĩa ResNet18Backbone với khởi tạo trọng số Kaiming/He (không load weights ImageNet)
# - Xuất ra vector đặc trưng (feature embedding) để truyền vào Multi-task Heads
