# Kiến trúc MobileNetV2 Backbone (Train từ đầu - KHÔNG dùng pretrained)
# Yêu cầu:
# - Tự định nghĩa InvertedResidual (Depthwise Separable Convolution + Linear Bottleneck)
# - Tự định nghĩa MobileNetV2Backbone với khởi tạo trọng số ngẫu nhiên (không load weights ImageNet)
# - Xuất ra vector đặc trưng (feature embedding) để truyền vào Multi-task Heads
