# Logic điều phối quá trình huấn luyện (Training Pipeline)
# Yêu cầu:
# - Khởi tạo DataLoader (Train/Val), Model (from scratch), Optimizer, Scheduler, Multi-task Loss
# - Hỗ trợ resume từ checkpoint nếu truyền --resume
# - Chạy vòng lặp 20 epochs (gọi train_one_epoch và validate trong engine.py)
# - Lưu checkpoint sau epoch 20 và best_model theo validation metric
# - Lưu training history vào results/<model_name>/
