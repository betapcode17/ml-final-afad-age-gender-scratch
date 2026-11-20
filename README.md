# AFAD-Age-Gender: Dự đoán Tuổi và Giới tính từ Ảnh Khuôn mặt (Multi-task CNN)

Dự án học máy **train từ đầu (from scratch, không sử dụng pretrained weights)** nhằm dự đoán đồng thời **Giới tính** (Classification — đầu ra là **chuỗi xác suất**) và **Tuổi** (Regression) từ ảnh khuôn mặt trên tập dữ liệu **AFAD-Full**.

---

## 1. Mục tiêu & Bài toán

- **Đầu vào (Input):** Ảnh khuôn mặt RGB từ bộ dữ liệu [AFAD-Full (Asian Face Age Dataset)](https://www.kaggle.com/datasets/lyk1652/afad-full).
- **Kiến trúc Multi-task CNN:** Dùng chung Backbone (train from scratch) và tách ra 2 nhánh đầu ra:
  1. **Nhánh Giới tính (Classification):** Đầu ra là **chuỗi xác suất** `[P(Nam), P(Nữ)]` (qua hàm Softmax), sử dụng `CrossEntropyLoss`.
  2. **Nhánh Tuổi (Regression):** Đầu ra là giá trị tuổi dự đoán liên tục, sử dụng `SmoothL1Loss` / `L1Loss`.
- **2 Mô hình đối chứng (Train from scratch):**
  - `ResNet-18` (Residual Network — đại diện kiến trúc chuẩn, hội tụ ổn định).
  - `MobileNetV2` (Inverted Residual + Depthwise Separable Conv — đại diện kiến trúc nhẹ, tối ưu tốc độ).

---

## 2. Cấu trúc Thư mục

```text
FINAL-PROJECT/
├── configs/
│   ├── resnet18.yaml               # Siêu tham số cho ResNet-18
│   └── mobilenetv2.yaml            # Siêu tham số cho MobileNetV2
│
├── data/
│   ├── raw/                        # Dataset AFAD gốc (không push lên Git)
│   ├── processed/                  # Dataset sau xử lý
│   └── splits/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── notebooks/
│   └── 01_dataset_analysis.ipynb   # Phân tích dữ liệu AFAD
│
├── src/
│   ├── datasets/
│   │   ├── afad_dataset.py         # Đọc ảnh và nhãn
│   │   └── transforms.py           # Data Augmentation
│   │
│   ├── models/
│   │   ├── resnet18.py             # ResNet-18 (from scratch)
│   │   ├── mobilenetv2.py          # MobileNetV2 (from scratch)
│   │   └── multitask.py            # Age Head + Gender Head
│   │
│   ├── training/
│   │   ├── train.py                # Training loop
│   │   ├── engine.py               # Train/Validation từng epoch
│   │   └── losses.py               # Multi-task loss
│   │
│   ├── evaluation/
│   │   ├── evaluate.py             # Đánh giá mô hình
│   │   ├── metrics.py              # MAE, RMSE, CS(5), Acc, F1, AUC
│   │   └── visualize.py            # Biểu đồ, confusion matrix
│   │
│   └── utils/
│       ├── seed.py
│       ├── logger.py
│       └── checkpoint.py
│
├── checkpoints/                    # Model đã train (mốc 20 epochs & best)
├── results/
│   ├── resnet18/                   # Logs, metrics & biểu đồ của ResNet-18
│   └── mobilenetv2/                # Logs, metrics & biểu đồ của MobileNetV2
│
├── app.py                          # Demo Streamlit (xuất chuỗi xác suất + tuổi)
├── train.py                        # File chạy training (hỗ trợ argparse)
├── evaluate.py                     # File chạy evaluation (hỗ trợ argparse)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 3. Yêu cầu Kỹ thuật Triển khai

- **Không sử dụng Pretrained:** Toàn bộ trọng số mô hình được khởi tạo ngẫu nhiên và huấn luyện từ đầu.
- **CLI với `argparse`:** Hỗ trợ truyền tham số linh hoạt (`--config`, `--data-dir`, `--model`, `--epochs`, `--batch-size`, `--lr`, `--img-size`, `--augment`, `--save-dir`, `--resume`).
- **Data Augmentation:** Tăng cường dữ liệu trên tập Train (Random Horizontal Flip, Random Rotation, Color Jitter...).
- **Checkpoint 20 Epochs:** Huấn luyện 20 epochs đầu tiên và lưu lại checkpoint để đánh giá hoặc train tiếp (`--resume`).
- **Đầu ra Chuỗi Xác Suất:** Trả về vector xác suất giới tính (ví dụ `[0.87, 0.13]`) kèm tuổi dự đoán, không xuất ra ảnh làm đầu ra.

---

## 4. Bộ Chỉ số Đánh giá

| Nhiệm vụ | Chỉ số | Ý nghĩa |
| :--- | :--- | :--- |
| **Giới tính (Classification)** | `Accuracy`, `Macro F1-score`, `ROC-AUC`, `Confusion Matrix` | Đánh giá độ chính xác phân loại và chất lượng chuỗi xác suất đầu ra |
| **Tuổi (Regression)** | `MAE`, `RMSE`, `CS(5)` (Cumulative Score $\le 5$ tuổi) | Đánh giá sai số tuyệt đối trung bình và tỷ lệ dự đoán lệch trong ngưỡng 5 năm |
| **Hệ thống** | `Parameters (M)`, `Model Size (MB)`, `Training Time`, `Inference (ms)` | So sánh hiệu năng giữa `ResNet-18` và `MobileNetV2` |

---

## 5. Hướng dẫn Sử dụng (Dự kiến)

```bash
# 1. Cài đặt thư viện
pip install -r requirements.txt

# 2. Huấn luyện mô hình từ đầu (20 epochs) và lưu checkpoint
python train.py --config configs/resnet18.yaml --data-dir ./data/raw/AFAD-Full --model resnet18 --epochs 20 --augment

# 3. Đánh giá trên tập Test
python evaluate.py --model resnet18 --checkpoint ./checkpoints/resnet18_epoch_20.pth --data-dir ./data/raw/AFAD-Full

# 4. Chạy Demo Streamlit (hiển thị chuỗi xác suất giới tính + tuổi dự đoán)
streamlit run app.py
```
