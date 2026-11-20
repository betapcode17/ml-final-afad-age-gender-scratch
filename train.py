# Entry point chạy quá trình huấn luyện (Training)
# Yêu cầu:
# - Sử dụng argparse để truyền tham số dòng lệnh:
#   --config, --data-dir, --model (resnet18 | mobilenetv2), --epochs (mặc định 20),
#   --batch-size, --lr, --img-size, --augment, --save-dir, --resume
# - Khởi tạo seed cố định để tái lập kết quả
# - Gọi pipeline từ src/training/train.py
# - Chạy 20 epochs đầu tiên và lưu checkpoint xuống thư mục checkpoints/
