import json
import os
from torchvision import transforms

DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


def load_norm_stats(stats_path="dataset/metadata/norm_stats.json"):
    """Đọc Mean & Std đã tính trên tập Train (fallback ImageNet nếu chưa có file)."""
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("mean", DEFAULT_MEAN), data.get("std", DEFAULT_STD)
    return DEFAULT_MEAN, DEFAULT_STD


def get_transforms(img_size=224, augment=True, vflip=False, stats_path="dataset/metadata/norm_stats.json"):
    mean, std = load_norm_stats(stats_path)
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    if not augment:
        return val_tf, val_tf

    aug_list = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ]
    if vflip:
        aug_list.append(transforms.RandomVerticalFlip(p=0.5))
    aug_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transforms.Compose(aug_list), val_tf
