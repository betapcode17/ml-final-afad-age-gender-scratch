# Hướng dẫn Thêm Model mới cho Thành viên nhóm (Template Guide)

Thư mục `notebooks/resnet18/` được thiết kế làm **template chuẩn** để các thành viên khác trong nhóm có thể tự thêm mô hình mới (ví dụ: `mobilenetv2`, `densenet121`, `vgg16`, ...) mà **không cần viết lại** code `Dataset`, `Transforms`, `Loss`, `Training Loop`, `Evaluation` hay `Streamlit App`.

---

## Quy trình 5 bước khi thêm một Model mới (ví dụ: `mobilenetv2`)

### Bước 1: Tạo file kiến trúc Backbone trong `src/models/`
Tạo file `src/models/mobilenetv2.py` (train **từ đầu**, không dùng `pretrained=True`).
Class Backbone chỉ cần thỏa mãn **2 điều kiện**:
1. Có thuộc tính `self.out_features` (ví dụ `1280` hoặc `512`).
2. Hàm `forward(self, x)` trả về vector đặc trưng shape `(B, self.out_features)`.

```python
class MobileNetV2Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        ...
        self.out_features = 1280

    def forward(self, x):
        ...
        return x.flatten(1)  # (B, 1280)
```

### Bước 2: Đăng ký model vào `src/models/builder.py`
Mở `src/models/builder.py`, import class vừa tạo và thêm 1 dòng vào `MODEL_REGISTRY`:

```python
from src.models.mobilenetv2 import MobileNetV2Backbone

MODEL_REGISTRY = {
    "resnet18": ResNet18Backbone,
    "mobilenetv2": MobileNetV2Backbone,  # <-- Thêm dòng này
}
```
> Ngay sau khi thêm vào `MODEL_REGISTRY`, các file `train.py`, `evaluate.py` và `app.py` sẽ **tự động nhận** `--model mobilenetv2`.

### Bước 3: Tạo file cấu hình & thư mục kết quả
- Tạo file config: `configs/mobilenetv2.yaml` (copy từ `configs/resnet18.yaml` và đổi `model: mobilenetv2`).
- Tạo thư mục lưu kết quả: `results/mobilenetv2/` (hoặc hệ thống sẽ tự tạo khi chạy `train.py` / `evaluate.py`).

### Bước 4: Tạo thư mục Notebook riêng theo template `notebooks/resnet18/`
1. Copy thư mục `notebooks/resnet18/` thành `notebooks/mobilenetv2/` (hoặc `notebooks/<tên_model>/`).
2. Trong notebook train mới:
   - Đổi lệnh train thành:
     ```bash
     !python train.py --model mobilenetv2 --config configs/mobilenetv2.yaml --splits-dir "{ACTIVE_SPLITS_DIR}" --epochs 20 --augment
     ```
   - Đổi lệnh evaluate thành:
     ```bash
     !python evaluate.py --model mobilenetv2 --splits-dir "{ACTIVE_SPLITS_DIR}" --checkpoint checkpoints/mobilenetv2_best.pth
     ```
   - Checkpoint sẽ tự động lưu thành `checkpoints/mobilenetv2_best.pth` & `checkpoints/mobilenetv2_epoch_20.pth`, còn metrics & biểu đồ lưu vào `results/mobilenetv2/`.

### Bước 5: Ghi chú về `data/splits/`
- Tập `data/splits/train.csv`, `val.csv`, `test.csv` đã được chia cố định (seed `42`).
- **Tất cả các mô hình phải dùng chung bộ splits này** để đảm bảo so sánh kết quả trên cùng một tập Test là công bằng.
