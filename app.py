# Ứng dụng Demo Streamlit dự đoán Tuổi và Giới tính từ ảnh khuôn mặt
# Yêu cầu:
# - Upload ảnh khuôn mặt đầu vào
# - Chọn mô hình (ResNet-18 hoặc MobileNetV2) và load checkpoint đã train từ đầu
# - Đầu ra BẮT BUỘC hiển thị CHUỖI XÁC XUẤT giới tính [P(Nam), P(Nữ)] (ví dụ Softmax probabilities)
#   kèm giá trị tuổi dự đoán (Regression), KHÔNG xuất ra ảnh làm đầu ra chính
