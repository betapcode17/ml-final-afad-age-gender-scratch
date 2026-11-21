# Ứng dụng Demo Streamlit dự đoán Nhóm tuổi (9 nhóm) và Giới tính (1 xác suất (0,1))
# Hỗ trợ tải ảnh từ máy tính HOẶC dán link ảnh trực tiếp từ trên mạng (URL)

import io
import urllib.request
import streamlit as st
import torch
from PIL import Image
from src.datasets.afad_dataset import AGE_GROUPS_9
from src.datasets.transforms import get_transforms
from src.models.builder import build_model, get_available_models

st.title("Dự đoán Nhóm Tuổi & Giới tính Người Châu Á (Multi-task ResNet-18)")
model_name = st.selectbox("Chọn mô hình", get_available_models())
age_mode = st.selectbox(
    "Chọn kiểu nhánh Tuổi",
    ["coral", "multiclass"],
    format_func=lambda m: "Kiểu B: CORAL Ordinal (8 ngưỡng -> 9 nhóm)" if m == "coral" else "Kiểu A: Multi-class (9 lớp Softmax)",
)
default_ckpt = f"checkpoints/{model_name}_{'typeB' if age_mode == 'coral' else 'typeA'}_best.pth"
ckpt_path = st.text_input("Đường dẫn Checkpoint", default_ckpt)

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
    g_prob, age_grp_idx, age_scores = model.predict_proba(tf(img).unsqueeze(0))
    p_male = float(g_prob.item())
    p_female = 1.0 - p_male
    pred_age_grp = AGE_GROUPS_9[int(age_grp_idx.item())]

    st.subheader("Kết quả dự đoán (1 Xác suất Giới tính (0,1) + 1 trong 9 Nhóm tuổi)")
    st.json({
        "model": model_name,
        "age_mode": age_mode,
        "gender_sigmoid_prob_male": round(p_male, 4),
        "gender_probabilities": {"female (0)": round(p_female, 4), "male (1)": round(p_male, 4)},
        "predicted_gender": "Male (1)" if p_male >= 0.5 else "Female (0)",
        "predicted_age_group": pred_age_grp,
    })
