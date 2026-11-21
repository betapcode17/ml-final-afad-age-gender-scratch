# Dataset: Asian-Age-Gender (16.522 ảnh Châu Á hợp nhất)

## 1. Phạm vi & Quy ước
- **Nhóm sắc tộc:** Chỉ tập trung vào người **Châu Á** (`Asian`: `12.243`, `East Asian`: `2.214`, `Southeast Asian`: `2.065`).
- **Giới tính:** `0 = Female (Nữ)` (`8.028` ảnh — `48.59%`), `1 = Male (Nam)` (`8.494` ảnh — `51.41%`).
- **9 Nhóm tuổi (`age_group`):** `0-2`, `3-9`, `10-19`, `20-29`, `30-39`, `40-49`, `50-59`, `60-69`, `70+`.

---

## 2. Chi tiết 4 Dataset Nguồn
1. **AFAD-Full (`165.501` ảnh thô → `8.318` ảnh sau gộp):** Ảnh khuôn mặt Châu Á tuổi `15–72` (tập trung `15–40`), cấu trúc `<age>/<111|112>/*.jpg`.
2. **AAF — All-Age-Faces (`13.322` ảnh thô → `2.806` ảnh sau gộp):** Ảnh khuôn mặt Châu Á đã căn chỉnh (`aglined faces`), tuổi `2–80`, tên file `%05dA%02d` (`ID <= 7380` là Female `0`, `ID > 7380` là Male `1`).
3. **UTKFace (`23.705` ảnh hợp lệ → `1.119` ảnh sau lọc Asian & gộp):** Ảnh đa sắc tộc tuổi `0–116`, tên file `[age]_[gender]_[race]_[date].jpg`, chỉ giữ `race == 2 (Asian)`.
4. **FairFace (`97.698` ảnh thô → `4.279` ảnh sau lọc Asian & gộp):** Ảnh đa sắc tộc 9 nhóm tuổi, chỉ giữ `East Asian` và `Southeast Asian`, ánh xạ trung điểm khoảng tuổi (`1.0, 6.0, 14.5, ..., 75.0`).

---

## 3. Dữ liệu Sau khi Gộp & Cấu trúc Thư mục
Từ **300.226** bản ghi thô → loại **1.220** ảnh trùng MD5 → còn **299.006** ảnh sạch → lọc Châu Á & cân bằng **Flexible Cap (`2.000`)** thu được **16.522** ảnh sạch chia chống rò rỉ (`70% / 15% / 15%`):

- `raw/`: Dữ liệu gốc ban đầu (`UTKFace`, `AAF`, `FairFace`).
- `interim/`: Dữ liệu trung gian (`cleaned/`, `resized/`).
- `processed/`: Toàn bộ **16.522 ảnh** chia sẵn theo `train/` (**11.565**), `val/` (**2.478**), `test/` (**2.479**) và phân cấp `<age_group>/<Female|Male>/`.
- `external/`: Dữ liệu bổ sung ngoài (nếu có).
- `metadata/`: Chứa `train.csv`, `val.csv`, `test.csv`, `split_info.csv`, `class_mapping.json`, `dataset_info.json`, `norm_stats.json`, `age_scaler.json`, `metadata_balanced.csv`, `metadata_merged_raw.csv`.
