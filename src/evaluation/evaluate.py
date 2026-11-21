# Đánh giá mô hình trên tập Test
# Tự động build đúng kiến trúc theo args.model và lưu kết quả vào results/<args.model>/

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
    plot_age_scatter,
    plot_confusion_matrix,
    plot_training_curves,
)
from src.models.builder import build_model
from src.utils.checkpoint import load_checkpoint


def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_tf = get_transforms(args.img_size, augment=False)
    test_ds = AFADDataset(os.path.join(args.splits_dir, "test.csv"), val_tf)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(args.model).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    all_probs, all_g_true, all_a_pred, all_a_true = [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for imgs, genders, ages in test_loader:
            probs, age_pred = model.predict_proba(imgs.to(device))
            all_probs.append(probs.cpu().numpy())
            all_g_true.append(genders.numpy())
            all_a_pred.append(age_pred.cpu().numpy())
            all_a_true.append(ages.numpy())
    infer_ms = (time.time() - t0) * 1000.0 / len(test_ds)

    gender_probs = np.concatenate(all_probs)
    gender_true = np.concatenate(all_g_true)
    age_pred = np.concatenate(all_a_pred)
    age_true = np.concatenate(all_a_true)

    metrics = compute_metrics(gender_probs, gender_true, age_pred, age_true)
    cm = metrics.pop("confusion_matrix")
    metrics["model"] = args.model
    metrics["inference_ms_per_img"] = round(infer_ms, 3)
    metrics["params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)

    out_dir = os.path.join("results", args.model)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion_matrix(cm, os.path.join(out_dir, "confusion_matrix.png"), model_name=args.model)
    plot_age_scatter(age_true, age_pred, os.path.join(out_dir, "age_scatter.png"), model_name=args.model)
    hist_csv = os.path.join(out_dir, "history.csv")
    if os.path.exists(hist_csv):
        plot_training_curves(hist_csv, out_dir)
    print(json.dumps(metrics, indent=2))
