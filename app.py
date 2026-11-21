# Ứng dụng Demo Streamlit dự đoán Tuổi và Giới tính từ ảnh khuôn mặt
# Tự động liệt kê mọi mô hình đã đăng ký trong src/models/builder.py

import streamlit as st
import torch
from PIL import Image
from src.datasets.transforms import get_transforms
from src.models.builder import build_model, get_available_models

st.title("Dự đoán Tuổi & Giới tính (AFAD — Multi-task CNN)")
model_name = st.selectbox("Chọn mô hình", get_available_models())
ckpt_path = st.text_input("Đường dẫn Checkpoint", f"checkpoints/{model_name}_epoch_20.pth")
uploaded = st.file_uploader("Tải ảnh khuôn mặt", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, width=220)

    model = build_model(model_name)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, tf = get_transforms(224, augment=False)
    probs, age = model.predict_proba(tf(img).unsqueeze(0))
    p = probs.squeeze().tolist()

    st.subheader("Kết quả dự đoán (Chuỗi xác suất + Tuổi)")
    st.json({
        "model": model_name,
        "gender_probabilities": {"female": round(p[0], 4), "male": round(p[1], 4)},
        "predicted_gender": "Female" if p[0] >= p[1] else "Male",
        "predicted_age": round(age.item(), 1),
    })
