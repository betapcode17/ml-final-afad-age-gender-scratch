# Trực quan hóa kết quả (Visualization)

import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_training_curves(history_csv, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, title in zip(
        axes,
        [("tr_loss", "val_loss"), ("tr_acc", "val_acc"), ("tr_mae", "val_mae")],
        ["Total Loss", "Gender Accuracy", "Age MAE"],
    ):
        ax.plot(df["epoch"], df[col[0]], label="Train")
        ax.plot(df["epoch"], df[col[1]], label="Val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()


def plot_confusion_matrix(cm, save_path, model_name="Model"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Female (0)", "Male (1)"],
        yticklabels=["Female (0)", "Male (1)"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{model_name} — Gender Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_age_scatter(age_true, age_pred, save_path, model_name="Model"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.scatter(age_true, age_pred, alpha=0.2, s=8)
    lims = [min(age_true.min(), age_pred.min()), max(age_true.max(), age_pred.max())]
    plt.plot(lims, lims, "r--")
    plt.xlabel("True Age")
    plt.ylabel("Predicted Age")
    plt.title(f"{model_name} — Age Prediction Scatter")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
