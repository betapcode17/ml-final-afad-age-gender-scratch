# Model Registry & Builder (Template mở rộng cho nhiều mô hình)
# Khi thành viên mới thêm mô hình (ví dụ: mobilenetv2.py, densenet121.py):
# 1. Import class Backbone mới tại đây (Backbone chỉ cần có thuộc tính `self.out_features`)
# 2. Đăng ký vào từ điển MODEL_REGISTRY bên dưới

from src.models.multitask import MultiTaskAgeGender
from src.models.resnet18 import ResNet18Backbone

MODEL_REGISTRY = {
    "resnet18": ResNet18Backbone,
    # "mobilenetv2": MobileNetV2Backbone,
    # "densenet121": DenseNet121Backbone,
}


def get_available_models():
    """Trả về danh sách tên các mô hình đã đăng ký."""
    return list(MODEL_REGISTRY.keys())


def build_model(model_name, num_genders=2, dropout=0.3):
    """Khởi tạo MultiTaskAgeGender từ tên mô hình truyền qua --model."""
    key = model_name.lower()
    if key not in MODEL_REGISTRY:
        available = ", ".join(get_available_models())
        raise ValueError(
            f"Mô hình '{model_name}' chưa được đăng ký trong MODEL_REGISTRY. "
            f"Các mô hình hiện có: [{available}]"
        )
    backbone = MODEL_REGISTRY[key]()
    return MultiTaskAgeGender(backbone, num_genders=num_genders, dropout=dropout)
