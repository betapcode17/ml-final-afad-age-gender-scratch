# Model Registry & Builder (Template mở rộng cho nhiều mô hình)
# Hỗ trợ resnet18, resnet18_se, resnet18_cbam_ms và age_mode in {'multiclass', 'coral', 'coral_v2', 'coral_v3'}.

from src.models.multitask import MultiTaskAgeGender
from src.models.resnet18 import ResNet18Backbone, ResNet18CustomBackbone


def _build_resnet18():
    return ResNet18Backbone(use_se=False)


def _build_resnet18_se():
    return ResNet18Backbone(use_se=True)


def _build_resnet18_cbam_ms():
    return ResNet18CustomBackbone()


MODEL_REGISTRY = {
    "resnet18": _build_resnet18,
    "resnet18_se": _build_resnet18_se,
    "resnet18_cbam_ms": _build_resnet18_cbam_ms,
}


def get_available_models():
    """Trả về danh sách tên các mô hình đã đăng ký."""
    return list(MODEL_REGISTRY.keys())


def build_model(model_name, age_mode="coral_v3", num_age_groups=9, dropout=0.3):
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
