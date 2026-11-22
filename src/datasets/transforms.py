import json
import os
import random
from PIL import Image, ImageOps
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


class PadToSquare:
    """Đệm ảnh chữ nhật (như AAF 309x398) thành hình vuông bằng màu trung bình tập Train
    trước khi Resize(224, 224) để tránh ép méo tỷ lệ khuôn mặt."""

    def __init__(self, fill=(136, 107, 95)):
        self.fill = tuple(int(c) for c in fill)

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        side = max(w, h)
        pad_l = (side - w) // 2
        pad_t = (side - h) // 2
        pad_r = side - w - pad_l
        pad_b = side - h - pad_t
        return ImageOps.expand(img, border=(pad_l, pad_t, pad_r, pad_b), fill=self.fill)


class RandomDownscaleUpscale:
    """Mô phỏng ảnh độ phân giải thấp (AFAD-Full 40-61px) để chống học đường tắt theo độ mờ."""

    def __init__(self, min_size=48, max_size=112, p=0.2):
        self.min_size = min_size
        self.max_size = max_size
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        w, h = img.size
        low = random.randint(self.min_size, self.max_size)
        return img.resize((low, low), Image.BILINEAR).resize((w, h), Image.BILINEAR)


def get_transforms(
    img_size=224,
    augment=True,
    vflip=False,
    stats_path="dataset/metadata/norm_stats.json",
    pad_square=True,
    ultimate_aug=False,
):
    """Tiền xử lý chuẩn hóa 224x224 theo norm_stats.json:
    - pad_square=True: đệm vuông tránh méo tỷ lệ ảnh AAF (309x398) cho cả Train/Val/Test
    - ultimate_aug=True: bật bộ Augmentation đầy đủ cho 02e (Resolution Jitter + RandomResizedCrop + Perspective + Blur)
    """
    mean, std = load_norm_stats(stats_path)
    fill_rgb = tuple(int(round(m * 255)) for m in mean)

    val_ops = []
    if pad_square:
        val_ops.append(PadToSquare(fill=fill_rgb))
    val_ops.extend([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_tf = transforms.Compose(val_ops)
    if not augment:
        return val_tf, val_tf

    aug_list = []
    if pad_square:
        aug_list.append(PadToSquare(fill=fill_rgb))

    if ultimate_aug:
        aug_list.extend([
            transforms.RandomResizedCrop(img_size, scale=(0.82, 1.0), ratio=(0.92, 1.08)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.92, 1.08)),
            transforms.RandomPerspective(distortion_scale=0.12, p=0.25),
            RandomDownscaleUpscale(min_size=48, max_size=112, p=0.2),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.15),
        ])
    else:
        aug_list.extend([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])

    if vflip:
        aug_list.append(transforms.RandomVerticalFlip(p=0.5))
    aug_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.10), value=0),
    ])
    return transforms.Compose(aug_list), val_tf
