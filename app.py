# Ứng dụng Demo Streamlit dự đoán Nhóm tuổi (9 nhóm) và Giới tính (1 xác suất (0,1))
# Hỗ trợ cả 02a (multiclass), 02b (coral), 02c (coral_v2) và 02d (coral_v3 + CBAM + MultiScale + ResNet-D)

import io
import urllib.request
import streamlit as st
import torch
from PIL import Image
from src.datasets.afad_dataset import AGE_GROUPS_9
from src.datasets.transforms import get_transforms
from src.models.builder import build_model, get_available_models

st.title("Dự đoán Nhóm Tuổi & Giới tính Người Châu Á (Multi-task ResNet-18)")
models_list = get_available_models()
default_m_idx = models_list.index("resnet18_hybrid") if "resnet18_hybrid" in models_list else 0
model_name = st.selectbox("Chọn kiến trúc Backbone", models_list, index=default_m_idx)
age_mode = st.selectbox(
    "Chọn kiểu nhánh Tuổi",
    ["coral_v4", "coral_v3", "coral_v2", "coral", "multiclass"],
    format_func=lambda m: {
        "coral_v4": "02e (Ultimate): Hybrid CBAM+MHSA + Gender->Age + Soft LDL + Calibrated",
        "coral_v3": "02d: CBAM + Multi-Scale + Weighted BCE + Phạt đơn điệu",
        "coral_v2": "02c (SE-CORAL v2): Shared CORAL + Focal",
        "coral": "02b (CORAL v1): 8 logits CORAL chuẩn",
        "multiclass": "02a (Baseline): Multi-class 9 lớp Softmax",
    }[m],
)
use_tta = st.checkbox("Bật TTA (Trung bình ảnh gốc + ảnh lật ngang)", value=True)
ckpt_defaults = {
    "coral_v4": "checkpoints/resnet18_typeE_ultimate_best_acc.pth",
    "coral_v3": "checkpoints/resnet18_typeD_cbam_ms_best.pth",
    "coral_v2": "checkpoints/resnet18_typeB_v2_best.pth",
    "coral": "checkpoints/resnet18_typeB_best.pth",
    "multiclass": "checkpoints/resnet18_typeA_best.pth",
}
ckpt_path = st.text_input("Đường dẫn Checkpoint", ckpt_defaults[age_mode])

img_url = st.text_input("Cách 1: Dán link ảnh khuôn mặt từ trên mạng (URL)", "")
uploaded = st.file_uploader("Cách 2: Hoặc tải file ảnh từ máy tính", type=["jpg", "jpeg", "png", "webp"])

img = None
if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
elif img_url.strip():
    try:
        req = urllib.request.Request(img_url.strip(), headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            img = Image.open(io.BytesIO(resp.read())).convert("RGB")
    except Exception as e:
        st.error(f"Không thể tải ảnh từ URL: {e}")

if img is not None:
    st.image(img, width=220)

    model = build_model(model_name, age_mode=age_mode)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, tf = get_transforms(224, augment=False)
    g_prob, age_grp_idx, age_scores = model.predict_proba(tf(img).unsqueeze(0), tta=use_tta)
    p_male = float(g_prob.item())
    p_female = 1.0 - p_male
    pred_age_grp = AGE_GROUPS_9[int(age_grp_idx.item())]

    st.subheader("Kết quả dự đoán (1 Xác suất Giới tính (0,1) + 1 trong 9 Nhóm tuổi)")
    st.json({
        "model": model_name,
        "age_mode": age_mode,
        "tta": use_tta,
        "gender_sigmoid_prob_male": round(p_male, 4),
        "gender_probabilities": {"female (0)": round(p_female, 4), "male (1)": round(p_male, 4)},
        "predicted_gender": "Male (1)" if p_male >= 0.5 else "Female (0)",
        "predicted_age_group": pred_age_grp,
    })
