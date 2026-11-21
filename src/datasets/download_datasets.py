"""Tải và giải nén UTKFace, AAF (All-Age-Faces) và FairFace vào data/raw/."""

import argparse
import os
import shutil
import tarfile
import zipfile
from pathlib import Path

import pandas as pd
import requests
from huggingface_hub import hf_hub_download
from tqdm import tqdm

AAF_DROPBOX_URL = "https://www.dropbox.com/s/a0lj1ddd54ns8qy/All-Age-Faces%20Dataset.zip?dl=1"

FAIRFACE_FILES = {
    "train_0": "0.25/train-00000-of-00002-d405faba4f4b9b85.parquet",
    "train_1": "0.25/train-00001-of-00002-dd3cb68164727418.parquet",
    "validation": "0.25/validation-00000-of-00001-951dbd63c8724ee1.parquet",
}

FAIRFACE_AGE_MAP = {
    0: "0-2",
    1: "3-9",
    2: "10-19",
    3: "20-29",
    4: "30-39",
    5: "40-49",
    6: "50-59",
    7: "60-69",
    8: "more than 70",
}
FAIRFACE_GENDER_MAP = {0: "Male", 1: "Female"}
FAIRFACE_RACE_MAP = {
    0: "East Asian",
    1: "Indian",
    2: "Black",
    3: "White",
    4: "Middle Eastern",
    5: "Latino_Hispanic",
    6: "Southeast Asian",
}


def download_file(url: str, dest: Path, desc: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=desc) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    return dest


def count_images(root: Path) -> int:
    if not root.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png"}
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def download_utkface(raw_dir: Path, cache_dir: Path, force: bool = False):
    out_dir = raw_dir / "UTKFace"
    if not force and count_images(out_dir) > 20000:
        print(f"[UTKFace] Đã có sẵn {count_images(out_dir):,} ảnh tại {out_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[UTKFace] Đang tải từ HuggingFace (py97/UTKFace-Cropped)...")
    tar_path = hf_hub_download(
        repo_id="py97/UTKFace-Cropped",
        filename="UTKFace.tar.gz",
        repo_type="dataset",
        cache_dir=str(cache_dir),
    )
    print(f"[UTKFace] Đang giải nén vào {out_dir}...")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(out_dir)
    print(f"[UTKFace] Hoàn tất: {count_images(out_dir):,} ảnh")


def download_aaf(raw_dir: Path, cache_dir: Path, force: bool = False):
    out_dir = raw_dir / "AAF"
    if not force and count_images(out_dir) > 10000:
        print(f"[AAF] Đã có sẵn {count_images(out_dir):,} ảnh tại {out_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "All-Age-Faces.zip"
    print("[AAF] Đang tải từ Dropbox (All-Age-Faces Dataset.zip)...")
    download_file(AAF_DROPBOX_URL, zip_path, desc="AAF.zip")
    print(f"[AAF] Đang giải nén vào {out_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    if zip_path.exists():
        zip_path.unlink()
    print(f"[AAF] Hoàn tất: {count_images(out_dir):,} ảnh")


def download_fairface(raw_dir: Path, cache_dir: Path, force: bool = False):
    out_dir = raw_dir / "FairFace"
    meta_csv = out_dir / "fairface_metadata.csv"
    imgs_dir = out_dir / "imgs"
    if not force and meta_csv.exists() and count_images(imgs_dir) > 90000:
        print(f"[FairFace] Đã có sẵn {count_images(imgs_dir):,} ảnh tại {out_dir}")
        return
    imgs_dir.mkdir(parents=True, exist_ok=True)

    records = []
    idx_global = 0
    for split_key, pq_rel in FAIRFACE_FILES.items():
        split_name = "train" if split_key.startswith("train") else "validation"
        print(f"[FairFace] Đang tải {pq_rel}...")
        pq_path = hf_hub_download(
            repo_id="HuggingFaceM4/FairFace",
            filename=pq_rel,
            repo_type="dataset",
            cache_dir=str(cache_dir),
        )
        df_pq = pd.read_parquet(pq_path)
        for row in tqdm(df_pq.itertuples(index=False), total=len(df_pq), desc=f"Extract {split_key}"):
            img_obj = row.image
            img_bytes = img_obj["bytes"] if isinstance(img_obj, dict) else img_obj
            fname = f"{split_name}_{idx_global:06d}.jpg"
            rel_file = f"imgs/{fname}"
            with open(imgs_dir / fname, "wb") as f:
                f.write(img_bytes)
            records.append({
                "file": rel_file,
                "age_group": FAIRFACE_AGE_MAP[int(row.age)],
                "gender": FAIRFACE_GENDER_MAP[int(row.gender)],
                "race": FAIRFACE_RACE_MAP[int(row.race)],
                "original_split": split_name,
            })
            idx_global += 1

    pd.DataFrame(records).to_csv(meta_csv, index=False)
    print(f"[FairFace] Hoàn tất: {len(records):,} ảnh & metadata tại {meta_csv}")


def main():
    p = argparse.ArgumentParser(description="Download UTKFace, AAF, and FairFace datasets")
    p.add_argument("--raw-dir", type=str, default="data/raw")
    p.add_argument("--force", action="store_true")
    p.add_argument("--keep-cache", action="store_true")
    args = p.parse_args()

    raw_dir = Path(args.raw_dir)
    cache_dir = raw_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    download_utkface(raw_dir, cache_dir, force=args.force)
    download_aaf(raw_dir, cache_dir, force=args.force)
    download_fairface(raw_dir, cache_dir, force=args.force)

    if not args.keep_cache and cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
        print("[Cleanup] Đã xóa cache tạm để tiết kiệm dung lượng đĩa.")


if __name__ == "__main__":
    main()
