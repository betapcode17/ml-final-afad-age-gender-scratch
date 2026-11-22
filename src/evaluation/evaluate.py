# Đánh giá mô hình trên tập Test cho Kiểu A (multiclass), Kiểu B (coral) và Kiểu B+ (coral_v2, hỗ trợ TTA)

import json
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.datasets.afad_dataset import AFADDataset
from src.datasets.transforms import get_transforms
from src.evaluation.metrics import compute_metrics
from src.evaluation.visualize import (
    plot_age_confusion_matrix,
    plot_confusion_matrix,
    plot_training_curves,
)
from src.models.builder import build_model
from src.utils.checkpoint import load_checkpoint


def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    age_mode = getattr(args, "age_mode", "coral_v2")
    use_tta = getattr(args, "tta", True)

    _, val_tf = get_transforms(args.img_size, augment=False)
    test_ds = AFADDataset(os.path.join(args.splits_dir, "test.csv"), val_tf)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args.model, age_mode=age_mode).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    all_g_prob, all_g_true, all_a_pred, all_a_true = [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for imgs, genders, age_groups in test_loader:
            g_prob, a_pred_grp, _ = model.predict_proba(imgs.to(device), tta=use_tta)
            all_g_prob.append(g_prob.cpu().numpy())
            all_g_true.append(genders.numpy())
            all_a_pred.append(a_pred_grp.cpu().numpy())
            all_a_true.append(age_groups.numpy())
    infer_ms = (time.time() - t0) * 1000.0 / len(test_ds)

    gender_prob = np.concatenate(all_g_prob)
    gender_true = np.concatenate(all_g_true)
    age_pred = np.concatenate(all_a_pred)
    age_true = np.concatenate(all_a_true)

    metrics = compute_metrics(gender_prob, gender_true, age_pred, age_true)
    cm_gender = metrics.pop("gender_confusion_matrix")
    cm_age = metrics.pop("age_confusion_matrix")
    tag = f"{args.model}_{age_mode}"
    metrics["model"] = tag
    metrics["age_mode"] = age_mode
    metrics["tta"] = use_tta
    metrics["inference_ms_per_img"] = round(infer_ms, 3)
    metrics["params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)

    out_dir = os.path.join("results", tag)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion_matrix(cm_gender, os.path.join(out_dir, "gender_confusion_matrix.png"), model_name=tag)
    plot_age_confusion_matrix(cm_age, os.path.join(out_dir, "age_confusion_matrix.png"), model_name=tag)
    hist_csv = os.path.join(out_dir, "history.csv")
    if os.path.exists(hist_csv):
        plot_training_curves(hist_csv, out_dir)
    print(json.dumps(metrics, indent=2))
