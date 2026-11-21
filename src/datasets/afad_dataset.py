import glob
import os
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
def create_afad_splits(data_root, splits_dir="data/splits", val_ratio=0.15, test_ratio=0.15, seed=42):
    os.makedirs(splits_dir, exist_ok=True)
    records = []
    for age_dir in os.listdir(data_root):
        if not age_dir.isdigit():
            continue
        age = float(age_dir)
        for g_code, g_label in [("111", 0), ("112", 1)]: # 0: Male, 1: Female
            for p in glob.glob(os.path.join(data_root, age_dir, g_code, "*.jpg")):
                if os.path.getsize(p) > 0:
                    records.append({"image_path": p.replace("\\", "/"), "gender": g_label, "age": age})
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
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = str(row["image_path"]).replace("\\", "/")
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        gender = torch.tensor(int(row["gender"]), dtype=torch.long)
        age = torch.tensor(float(row["age"]), dtype=torch.float32)
        return img, gender, age