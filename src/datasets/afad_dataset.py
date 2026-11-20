# Dataset loader cho tập AFAD-Full
# Yêu cầu:
# - Đọc danh sách ảnh và nhãn (tuổi, giới tính) từ data/splits/ (train.csv, val.csv, test.csv) hoặc quét trực tiếp từ data/raw/
# - Parse cấu trúc AFAD-Full/<age>/<gender_code>/*.jpg (111: Nam, 112: Nữ -> map về 0/1)
# - Trả về (image_tensor, gender_label, age_label)
