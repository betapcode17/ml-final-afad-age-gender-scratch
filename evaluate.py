# Entry point chạy quá trình đánh giá (Evaluation) trên tập Test
# Yêu cầu:
# - Sử dụng argparse (--model, --checkpoint, --data-dir, --batch-size, --output-dir)
# - Load checkpoint đã lưu từ thư mục checkpoints/
# - Gọi hàm đánh giá từ src/evaluation/evaluate.py
# - Xuất kết quả (Accuracy, F1, ROC-AUC, MAE, RMSE, CS(5), Confusion Matrix) vào results/<model>/
