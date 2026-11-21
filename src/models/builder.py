# Model Registry & Builder (Template mở rộng cho nhiều mô hình)
# Hỗ trợ age_mode='multiclass' (Kiểu A - 9 lớp) và age_mode='coral' (Kiểu B - CORAL 8 ngưỡng).

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


def build_model(model_name, age_mode="coral", num_age_groups=9, dropout=0.3):
    """Khởi tạo MultiTaskAgeGender từ tên mô hình truyền qua --model và --age-mode."""
    key = model_name.lower()
    if key not in MODEL_REGISTRY:
        available = ", ".join(get_available_models())
        raise ValueError(
            f"Mô hình '{model_name}' chưa được đăng ký trong MODEL_REGISTRY. "
            f"Các mô hình hiện có: [{available}]"
        )
    backbone = MODEL_REGISTRY[key]()
    return MultiTaskAgeGender(
        backbone,
        age_mode=age_mode,
        num_age_groups=num_age_groups,
        dropout=dropout,
    )
