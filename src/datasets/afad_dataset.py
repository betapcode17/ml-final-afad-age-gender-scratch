import glob
import os
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

AGE_GROUPS_9 = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
AGE_GROUP_TO_IDX = {g: i for i, g in enumerate(AGE_GROUPS_9)}
AGE_CUTOFFS_8 = [2.5, 9.5, 19.5, 29.5, 39.5, 49.5, 59.5, 69.5]


def map_age_to_group_idx(age_val: float) -> int:
    a = int(round(float(age_val)))
    if a <= 2:
        return 0
    if a <= 9:
        return 1
    if a <= 19:
        return 2
    if a <= 29:
        return 3
    if a <= 39:
        return 4
    if a <= 49:
        return 5
    if a <= 59:
        return 6
    if a <= 69:
        return 7
    return 8


def create_afad_splits(data_root, splits_dir="data/splits", val_ratio=0.15, test_ratio=0.15, seed=42):
    os.makedirs(splits_dir, exist_ok=True)
    records = []
    for age_dir in os.listdir(data_root):
        if not age_dir.isdigit():
            continue
        age = float(age_dir)
        age_grp = AGE_GROUPS_9[map_age_to_group_idx(age)]
        for g_code, g_label in [("112", 0), ("111", 1)]:
            for p in glob.glob(os.path.join(data_root, age_dir, g_code, "*.jpg")):
                if os.path.getsize(p) > 0:
                    records.append({
                        "image_path": p.replace("\\", "/"),
                        "gender": g_label,
                        "age": age,
                        "age_group": age_grp,
                    })
    df = pd.DataFrame(records)
    train_df, temp_df = train_test_split(
        df, test_size=(val_ratio + test_ratio), stratify=df["gender"], random_state=seed
    )
    rel_test = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=rel_test, stratify=temp_df["gender"], random_state=seed
    )
    train_df.to_csv(os.path.join(splits_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(splits_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(splits_dir, "test.csv"), index=False)
    return len(train_df), len(val_df), len(test_df)


class AFADDataset(Dataset):
    """Dataset khuôn mặt Châu Á:
    - Mặc định (return_exact_age=False): trả về (img, gender_float32, age_group_idx_0_to_8)
    - Khi return_exact_age=True (dùng cho 02e Soft Ordinal LDL):
      trả về thêm (age_exact_float32, has_exact_age_float32 [1.0 nếu != FairFace, 0.0 nếu == FairFace])
    """

    def __init__(self, csv_path, transform=None, img_root_override=None, return_exact_age=False):
        self.df = pd.read_csv(csv_path)
        if img_root_override:
            self.df["image_path"] = (
                self.df["image_path"]
                .astype(str)
                .str.replace("dataset/processed", img_root_override, regex=False)
            )
        self.transform = transform
        self.return_exact_age = return_exact_age

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = str(row["image_path"]).replace("\\", "/")
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        gender = torch.tensor(float(row["gender"]), dtype=torch.float32)
        if "age_group" in row and pd.notna(row["age_group"]):
            ag_idx = AGE_GROUP_TO_IDX[str(row["age_group"]).strip()]
        else:
            ag_idx = map_age_to_group_idx(float(row["age"]))
        age_group = torch.tensor(ag_idx, dtype=torch.long)

        if self.return_exact_age:
            age_exact = torch.tensor(float(row["age"]) if "age" in row and pd.notna(row["age"]) else 0.0, dtype=torch.float32)
            src = str(row["source"]).strip() if "source" in row and pd.notna(row["source"]) else ""
            has_exact = torch.tensor(0.0 if src == "FairFace" else 1.0, dtype=torch.float32)
            return img, gender, age_group, age_exact, has_exact
        return img, gender, age_group
