# Bộ chỉ số đánh giá (Evaluation Metrics) cho mô hình Multi-task:
# - Nhánh Giới tính (1 xác suất (0,1)): Accuracy, Precision, Recall, Macro F1-score, ROC-AUC, Confusion Matrix (2x2)
# - Nhánh Tuổi (9 nhóm tuổi 0..8): Accuracy (9 lớp), Adjacent Accuracy (±1 nhóm), Macro Precision/Recall/F1, Confusion Matrix (9x9)

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(gender_prob, gender_true, age_grp_pred, age_grp_true, num_age_groups=9):
    gender_true = np.asarray(gender_true).astype(int)
    gender_prob = np.asarray(gender_prob).astype(float).reshape(-1)
    gender_pred = (gender_prob >= 0.5).astype(int)

    age_grp_true = np.asarray(age_grp_true).astype(int)
    age_grp_pred = np.asarray(age_grp_pred).astype(int)

    return {
        "gender_acc": float(accuracy_score(gender_true, gender_pred)),
        "gender_precision_macro": float(precision_score(gender_true, gender_pred, average="macro", zero_division=0)),
        "gender_recall_macro": float(recall_score(gender_true, gender_pred, average="macro", zero_division=0)),
        "gender_f1_macro": float(f1_score(gender_true, gender_pred, average="macro", zero_division=0)),
        "gender_roc_auc": float(roc_auc_score(gender_true, gender_prob)),
        "age_group_acc": float(accuracy_score(age_grp_true, age_grp_pred)),
        "age_adjacent_acc": float(np.mean(np.abs(age_grp_pred - age_grp_true) <= 1)),
        "age_precision_macro": float(precision_score(age_grp_true, age_grp_pred, average="macro", zero_division=0)),
        "age_recall_macro": float(recall_score(age_grp_true, age_grp_pred, average="macro", zero_division=0)),
        "age_f1_macro": float(f1_score(age_grp_true, age_grp_pred, average="macro", zero_division=0)),
        "gender_confusion_matrix": confusion_matrix(gender_true, gender_pred, labels=[0, 1]),
        "age_confusion_matrix": confusion_matrix(age_grp_true, age_grp_pred, labels=list(range(num_age_groups))),
    }
