# Bộ chỉ số đánh giá (Evaluation Metrics)
# Yêu cầu:
# - Nhánh Giới tính (Classification): Accuracy, Precision, Recall, Macro F1-score, ROC-AUC (từ chuỗi xác suất), Confusion Matrix
# - Nhánh Tuổi (Regression): MAE, RMSE, CS(k) (Cumulative Score ở ngưỡng k=5 tuổi)
# - Hệ thống: Tổng số tham số (Params), kích thước checkpoint, thời gian suy luận (ms/ảnh)


import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error, roc_auc_score,
)
def compute_metrics(gender_probs, gender_true, age_pred, age_true, cs_threshold=5.0):
    gender_pred = np.argmax(gender_probs, axis=1)
    return {
        "gender_acc": float(accuracy_score(gender_true, gender_pred)),
        "gender_f1_macro": float(f1_score(gender_true, gender_pred, average="macro")),
        "gender_roc_auc": float(roc_auc_score(gender_true, gender_probs[:, 1])),
        "age_mae": float(mean_absolute_error(age_true, age_pred)),
        "age_rmse": float(np.sqrt(mean_squared_error(age_true, age_pred))),
        f"age_cs_{int(cs_threshold)}": float(np.mean(np.abs(age_pred - age_true) <= cs_threshold) * 100.0),
        "confusion_matrix": confusion_matrix(gender_true, gender_pred),
    }