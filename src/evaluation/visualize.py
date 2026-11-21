# Trực quan hóa kết quả (Visualization) cho bài toán Giới tính (1 xác suất) & 9 Nhóm tuổi

import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.datasets.afad_dataset import AGE_GROUPS_9


def plot_training_curves(history_csv, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(history_csv)

    # 1. Loss curves (Gender, Age, Total)
    if {"tr_gender_loss", "val_gender_loss", "tr_age_loss", "val_age_loss"}.issubset(df.columns):
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
        for ax, col, title in zip(
            axes,
            [("tr_gender_loss", "val_gender_loss"), ("tr_age_loss", "val_age_loss"), ("tr_loss", "val_loss")],
            ["Gender Loss (BCE)", "Age Loss", "Total Multi-task Loss"],
        ):
            ax.plot(df["epoch"], df[col[0]], marker="o", markersize=3, label="Train")
            ax.plot(df["epoch"], df[col[1]], marker="s", markersize=3, label="Validation")
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.3)
            ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "gender_age_loss_curves.png"), dpi=150)
        plt.close()

    # 2. Accuracy curves (Gender & Age Group)
    if {"tr_acc", "val_acc", "tr_age_grp_acc", "val_age_grp_acc"}.issubset(df.columns):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, col, title in zip(
            axes,
            [("tr_acc", "val_acc"), ("tr_age_grp_acc", "val_age_grp_acc")],
            ["Gender Accuracy", "Age-Group Accuracy (9 nhóm tuổi)"],
        ):
            ax.plot(df["epoch"], df[col[0]] * 100.0, marker="o", markersize=3, label="Train")
            ax.plot(df["epoch"], df[col[1]] * 100.0, marker="s", markersize=3, label="Validation")
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy (%)")
            ax.grid(alpha=0.3)
            ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "gender_age_accuracy_curves.png"), dpi=150)
        plt.close()


def plot_confusion_matrix(cm, save_path, model_name="Model"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(5.5, 4.5))
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
    plt.title(f"{model_name} — Ma trận nhầm lẫn Giới tính")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_age_confusion_matrix(cm_age, save_path, model_name="Model"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(9, 7.5))
    sns.heatmap(
        cm_age,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=AGE_GROUPS_9,
        yticklabels=AGE_GROUPS_9,
    )
    plt.xlabel("Predicted Age Group")
    plt.ylabel("True Age Group")
    plt.title(f"{model_name} — Ma trận nhầm lẫn Độ tuổi (9 nhóm)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
