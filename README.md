# Asian-Age-Gender: Dự đoán Tuổi và Giới tính từ Ảnh Khuôn mặt Châu Á (Multi-task CNN)

Dự án học máy **train từ đầu (from scratch, không sử dụng pretrained weights)** nhằm dự đoán đồng thời **Giới tính** (Classification — đầu ra là **chuỗi xác suất** `[P(Female), P(Male)]`) và **Tuổi** (Regression) từ ảnh khuôn mặt.

**Phạm vi dữ liệu:** Chỉ tập trung vào **nhóm sắc tộc Châu Á (Asian)** (`AFAD-Full`, `AAF`, `UTKFace[Asian]`, `FairFace[East Asian, Southeast Asian]`), không mở rộng sang các nhóm sắc tộc khác.

---

## 1. Cấu trúc Thư mục

```text
FINAL-PROJECT/
├── configs/
│   └── resnet18.yaml                  # Siêu tham số cho ResNet-18 (img_size: 224)
├── dataset/
│   ├── raw/                           # Dữ liệu gốc (UTKFace, AAF, FairFace)
│   ├── interim/                       # Dữ liệu trung gian (cleaned/, resized/)
│   ├── processed/                     # 16.522 ảnh Châu Á đã chia theo train/ val/ test/
│   │   ├── train/                     # <age_group>/<Female|Male>/ (11.565 ảnh)
│   │   ├── val/                       # <age_group>/<Female|Male>/ (2.478 ảnh)
│   │   └── test/                      # <age_group>/<Female|Male>/ (2.479 ảnh)
│   ├── external/                      # Dataset bổ sung ngoài (.gitkeep)
│   ├── metadata/                      # train.csv, val.csv, test.csv, split_info.csv, class_mapping.json, dataset_info.json, norm_stats.json, age_scaler.json
│   └── README.md                      # Mô tả chi tiết các dataset
├── notebooks/
│   ├── README.md                      # Hướng dẫn nhân bản template cho model mới
│   ├── 01_dataset_analysis.ipynb      # Phân tích dữ liệu hợp nhất, t-SNE, trực quan tiền xử lý & ZIP
│   └── resnet18/
│       └── 02_train_eval_resnet18.ipynb
├── src/
│   ├── datasets/                      # download_datasets.py, prepare_multidata.py, afad_dataset.py, transforms.py
│   ├── models/                        # builder.py (MODEL_REGISTRY), resnet18.py, multitask.py
│   ├── training/                      # train.py, engine.py, losses.py
│   ├── evaluation/                    # evaluate.py, metrics.py, visualize.py
│   └── utils/                         # seed.py, logger.py, checkpoint.py
├── checkpoints/                       # Lưu <model>_best.pth và <model>_epoch_20.pth
├── results/
│   ├── eda/                           # Biểu đồ phân bổ, t-SNE, Resize/Geometric, Norm/Augment
│   └── resnet18/                      # Logs, test_metrics.json & biểu đồ đánh giá
├── app.py                             # Demo Streamlit (xuất chuỗi xác suất + tuổi)
├── train.py
├── evaluate.py
└── requirements.txt
```

---

## 2. Mô tả Từng Dataset Nguồn & Dữ liệu Sau khi Gộp

### 2.1. Mô tả 4 Bộ Dữ liệu Nguồn

| Dataset | Quy mô thô / Hợp lệ | Dải tuổi & Định dạng nhãn | Chủng tộc gốc | Vai trò & Cách xử lý khi gộp |
| :--- | :---: | :--- | :--- | :--- |
| **1. AFAD-Full** *(Asian Face Age Dataset)* | **165.501** ảnh | Tuổi liên tục `15–72` (`15–40` chiếm `99.35%`). Thư mục `<age>/<111\|112>/*.jpg` (`111=Male`, `112=Female`). | 100% **Asian** | Nguồn xương sống cho nhóm tuổi `15–40`; ảnh vuông sẵn (`median 94px`). Sau cân bằng giữ **8.318 ảnh**. |
| **2. AAF** *(All-Age-Faces)* | **13.322** ảnh (`aglined faces`) | Tuổi liên tục `2–80`. Tên file `%05dA%02d.jpg`: `ID <= 7380` → **Female (`0`)** (`7.381` ảnh), `ID > 7380` → **Male (`1`)** (`5.941` ảnh). | Chủ yếu **Asian** | Bổ sung ảnh khuôn mặt Châu Á đã căn chỉnh ở dải tuổi rộng (`2–80`) và có `identity_id` để chia tập chống rò rỉ. Sau cân bằng giữ **2.806 ảnh**. |
| **3. UTKFace** *(Aligned & Cropped)* | **23.708** ảnh thô (**23.705** hợp lệ) | Tuổi liên tục `0–116`. Tên file `[age]_[gender]_[race]_[date].jpg` (`gender: 0=Male, 1=Female`, `race: 0..4`). | Đa sắc tộc (`White, Black, Asian, Indian, Others`) | Chỉ lọc giữ **`race == "Asian"`** (loại 4 chủng tộc còn lại); bổ sung tốt trẻ em `<15` và người lớn tuổi. Sau lọc & cân bằng giữ **1.119 ảnh**. |
| **4. FairFace** *(WACV 2021, config `0.25`)* | **97.698** ảnh (`86.744` train + `10.954` val) | 9 nhóm tuổi chuỗi (`0-2` … `more than 70` → chuẩn hóa thành `70+`), `gender: Male/Female`. | 7 nhóm chủng tộc | Chỉ lọc giữ **`East Asian`** và **`Southeast Asian`**; quy đổi khoảng tuổi sang trung điểm (`1.0, 6.0, 14.5, ..., 75.0`) để cân bằng các nhóm tuổi hiếm (`0-2`, `3-9`, `50-69`, `70+`). Sau lọc & cân bằng giữ **4.279 ảnh**. |

### 2.2. Quá trình Làm sạch, Lọc Châu Á & Cân bằng (`Flexible Cap = 2.000`)
- **Tổng bản ghi thô từ 4 nguồn:** **300.226** ảnh → Kiểm tra ảnh hỏng (`Image.verify()`): loại **0** ảnh hỏng → Khử trùng lặp `MD5`: loại **1.220** ảnh trùng → Còn **299.006** ảnh sạch đa sắc tộc.
- **Lọc phạm vi Châu Á & Chuẩn hóa nhãn:**
  - Chỉ giữ nhóm sắc tộc Châu Á (`Asian`: **12.243**, `East Asian`: **2.214**, `Southeast Asian`: **2.065**).
  - Chuẩn hóa biến tuổi về **9 nhóm `age_group`** (`0-2`, `3-9`, `10-19`, `20-29`, `30-39`, `40-49`, `50-59`, `60-69`, `70+`).
  - Chuẩn hóa giới tính thống nhất toàn dự án: **`0 = Female (Nữ)`**, **`1 = Male (Nam)`**.
  - Áp dụng **Flexible Cap (`cap = 2.000`)** trên mỗi tổ hợp `(age_group, gender)`: giữ nguyên nhóm `< 2.000` và lấy mẫu ngẫu nhiên nhóm `> 2.000`.

### 2.3. Tổng hợp Dữ liệu Cuối cùng Sau khi Gộp (`16.522` ảnh Châu Á)
- **Tổng số ảnh sạch sau gộp & lọc Châu Á:** **16.522 ảnh**:
  - Theo nguồn: `AFAD-Full`: **8.318** (`50.35%`) | `FairFace`: **4.279** (`25.90%`) | `AAF`: **2.806** (`16.98%`) | `UTKFace`: **1.119** (`6.77%`).
  - Theo giới tính: **8.028 Female (`0`)** [`48.59%`] | **8.494 Male (`1`)** [`51.41%`] (tỷ lệ gần 1:1).
  - Dải tuổi số: `1.0 → 110.0` tuổi (trung bình tập Train: `30.80 ± 20.78`).

| Nhóm tuổi (`age_group`) | Trước lọc & Cap — Đa sắc tộc (Nữ / Nam / Tổng) | Sau lọc Châu Á & Cap (Nữ `0` / Nam `1` / Tổng) |
| :--- | :---: | :---: |
| `0-2` | 1.562 / 2.178 / 3.740 | 568 / 844 / **1.412** |
| `3-9` | 6.564 / 7.472 / 14.036 | 664 / 681 / **1.345** |
| `10-19` | 19.928 / 17.271 / 37.199 | 1.505 / 1.586 / **3.091** |
| `20-29` | 60.417 / 76.658 / 137.075 | 1.517 / 1.698 / **3.215** |
| `30-39` | 25.590 / 44.329 / 69.919 | 1.297 / 1.460 / **2.757** |
| `40-49` | 6.836 / 11.540 / 18.376 | 749 / 680 / **1.429** |
| `50-59` | 3.723 / 6.879 / 10.602 | 579 / 464 / **1.043** |
| `60-69` | 2.010 / 3.204 / 5.214 | 672 / 547 / **1.219** |
| `70+` | 1.411 / 1.434 / 2.845 | 477 / 534 / **1.011** |
| **Tổng cộng** | **128.041 / 170.965 / 299.006** | **8.028 / 8.494 / 16.522** |

### 2.4. Chia tập Chống rò rỉ (`70% / 15% / 15%`) & Thông số Chuẩn hóa (`224×224`)
- **Chia tập (`seed = 42`, group-aware theo `identity_id` / `md5`, kiểm tra trùng path & MD5 giữa 3 tập = 0):**
  - `Train`: **11.565** ảnh | `Validation`: **2.478** ảnh | `Test`: **2.479** ảnh.
- **Mean & Std tính trên tập Train (`224 × 224` RGB, [norm_stats.json](dataset/metadata/norm_stats.json)):**
  - `mean = [0.535029, 0.42123, 0.372826]`
  - `std  = [0.276233, 0.248417, 0.242609]`
- **Thông số chuẩn hóa tuổi ([age_scaler.json](dataset/metadata/age_scaler.json)):**
  - `train_age_mean = 30.8043`, `train_age_std = 20.7804`, `min = 1.0`, `max = 110.0`.
  - Z-score: `(age - 30.8043) / 20.7804` | Nghịch đảo: `age_norm * 20.7804 + 30.8043`.

### 2.5. Biểu đồ Trực quan hóa Dữ liệu & Tiền xử lý (`results/eda/`)
- **Phân bổ dữ liệu Giới tính & Độ tuổi (Trước vs Sau):** ![Phân bổ dữ liệu](results/eda/age_gender_distribution.png)
- **Biểu diễn t-SNE theo 9 Nhóm tuổi & Giới tính:** ![t-SNE](results/eda/tsne_age_gender.png)
- **Ảnh sau Tiền xử lý Resize (`224×224`) & Geometric (`Flip + Rotate`):** ![Resize & Geometric](results/eda/preprocess_resize_geometric.png)
- **Ảnh sau Chuẩn hóa (`Mean/Std`) & Data Augmentation:** ![Chuẩn hóa & Augmentation](results/eda/preprocess_norm_augment.png)

---

## 3. Hướng dẫn Chạy nhanh

```bash
# 1. Cài đặt thư viện
pip install -r requirements.txt

# 2. Huấn luyện từ đầu 20 epochs (img_size 224x224)
python train.py --model resnet18 --config configs/resnet18.yaml --epochs 20 --batch-size 64 --lr 1e-3 --img-size 224 --augment

# 3. Đánh giá trên tập Test
python evaluate.py --model resnet18 --checkpoint checkpoints/resnet18_best.pth --img-size 224

# 4. Chạy Demo Streamlit
streamlit run app.py
```

---

## 4. Hướng dẫn cho Thành viên nhóm khi Thêm Model mới
Xem hướng dẫn tại [notebooks/README.md](notebooks/README.md).
