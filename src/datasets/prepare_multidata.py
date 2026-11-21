"""Hợp nhất đa nguồn (AFAD-Full + UTKFace + FairFace + AAF), chuẩn hóa 9 nhóm tuổi,
khử trùng lặp/ảnh lỗi, cân bằng Flexible Cap (2000), chia tập chống rò rỉ,
và tính Mean/Std trên tập Train (224x224) + công thức chuẩn hóa nhãn Age."""

import argparse
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

AGE_GROUPS_9 = [
    "0-2",
    "3-9",
    "10-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "70+",
]

AGE_GROUP_MIDPOINT = {
    "0-2": 1.0,
    "3-9": 6.0,
    "10-19": 14.5,
    "20-29": 24.5,
    "30-39": 34.5,
    "40-49": 44.5,
    "50-59": 54.5,
    "60-69": 64.5,
    "70+": 75.0,
}

ASIAN_RACES = {"Asian", "East Asian", "Southeast Asian"}

UTK_RACE_MAP = {
    0: "White",
    1: "Black",
    2: "Asian",
    3: "Indian",
    4: "Others",
}

AAF_PATTERN = re.compile(r"^(\d{5})A(\d{2})$", re.IGNORECASE)
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def map_age_to_9_groups(age: float) -> str:
    """Chuẩn hóa tuổi số về đúng 9 chuỗi age_group tương thích FairFace."""
    a = int(round(float(age)))
    if a <= 2:
        return "0-2"
    if a <= 9:
        return "3-9"
    if a <= 19:
        return "10-19"
    if a <= 29:
        return "20-29"
    if a <= 39:
        return "30-39"
    if a <= 49:
        return "40-49"
    if a <= 59:
        return "50-59"
    if a <= 69:
        return "60-69"
    return "70+"


def find_img_in_all_subdirs(root_dir: str):
    """Duyệt đệ quy toàn bộ thư mục con tìm ảnh (.jpg/.jpeg/.png) và tạo bảng ánh xạ đường dẫn."""
    all_paths = []
    by_rel = {}
    by_name = {}
    root_abs = os.path.abspath(root_dir)
    if not os.path.isdir(root_abs):
        return all_paths, by_rel, by_name

    for dirpath, _, filenames in os.walk(root_abs):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                full_p = os.path.join(dirpath, fn).replace("\\", "/")
                rel_p = os.path.relpath(full_p, root_abs).replace("\\", "/")
                all_paths.append(full_p)
                by_rel[rel_p] = full_p
                by_name[fn] = full_p
    return all_paths, by_rel, by_name


def parse_afad(afad_dir: str) -> pd.DataFrame:
    paths, _, _ = find_img_in_all_subdirs(afad_dir)
    records = []
    for full_p in paths:
        parts = full_p.split("/")
        if len(parts) < 3:
            continue
        g_code = parts[-2]
        age_str = parts[-3]
        if not age_str.isdigit() or g_code not in ("111", "112"):
            continue
        age = float(age_str)
        if not (15 <= age <= 72):
            continue
        gender_str = "Male" if g_code == "111" else "Female"
        gender_label = 0 if gender_str == "Female" else 1
        records.append({
            "file": os.path.basename(full_p),
            "image_path": full_p,
            "age": age,
            "age_group": str(map_age_to_9_groups(age)),
            "gender": gender_str,
            "gender_label": gender_label,
            "race": "Asian",
            "source": "AFAD-Full",
            "identity_id": "",
        })
    return pd.DataFrame(records)


def parse_aaf(aaf_dir: str) -> pd.DataFrame:
    # Ưu tiên thư mục aligned faces (tác giả AAF đặt tên "aglined faces")
    aligned_dir = None
    for dirpath, dirnames, _ in os.walk(aaf_dir):
        for d in dirnames:
            dl = d.lower()
            if "aligned" in dl or "aglined" in dl:
                aligned_dir = os.path.join(dirpath, d)
                break
        if aligned_dir:
            break

    scan_dir = aligned_dir if aligned_dir else aaf_dir
    print(f"[Parse AAF] Scanning directory: {scan_dir}")
    paths, _, _ = find_img_in_all_subdirs(scan_dir)
    records = []
    for full_p in paths:
        stem = os.path.splitext(os.path.basename(full_p))[0]
        m = AAF_PATTERN.match(stem)
        if not m:
            continue
        person_id = int(m.group(1))
        age = float(m.group(2))
        if not (0 <= age <= 100):
            continue
        # Quy ước AAF: ID <= 7380 là Nữ (Female - 0), ID > 7380 là Nam (Male - 1)
        if person_id <= 7380:
            gender_str = "Female"
            gender_label = 0
        else:
            gender_str = "Male"
            gender_label = 1
        records.append({
            "file": os.path.basename(full_p),
            "image_path": full_p,
            "age": age,
            "age_group": str(map_age_to_9_groups(age)),
            "gender": gender_str,
            "gender_label": gender_label,
            "race": "Asian",
            "source": "AAF",
            "identity_id": f"aaf_{person_id:05d}",
        })
    return pd.DataFrame(records)


def parse_utkface(utk_dir: str) -> pd.DataFrame:
    paths, _, _ = find_img_in_all_subdirs(utk_dir)
    records = []
    for full_p in paths:
        fn = os.path.basename(full_p)
        parts = fn.split("_")
        if len(parts) < 4:
            continue
        try:
            age = float(parts[0])
            g_raw = int(parts[1])
            r_raw = int(parts[2])
        except ValueError:
            continue
        if not (0 <= age <= 116) or g_raw not in (0, 1) or r_raw != 2:  # Chỉ giữ Asian (2)
            continue
        gender_str = "Male" if g_raw == 0 else "Female"
        gender_label = 0 if gender_str == "Female" else 1
        records.append({
            "file": fn,
            "image_path": full_p,
            "age": age,
            "age_group": str(map_age_to_9_groups(age)),
            "gender": gender_str,
            "gender_label": gender_label,
            "race": UTK_RACE_MAP[r_raw],
            "source": "UTKFace",
            "identity_id": "",
        })
    return pd.DataFrame(records)


def parse_fairface(fairface_dir: str) -> pd.DataFrame:
    meta_csv = os.path.join(fairface_dir, "fairface_metadata.csv")
    if not os.path.exists(meta_csv):
        return pd.DataFrame()
    _, by_rel, by_name = find_img_in_all_subdirs(fairface_dir)
    df_meta = pd.read_csv(meta_csv)
    records = []
    for row in df_meta.itertuples(index=False):
        rel = str(row.file).replace("\\", "/")
        full_p = by_rel.get(rel) or by_name.get(os.path.basename(rel))
        if not full_p:
            continue
        ag = str(row.age_group).strip()
        if ag == "more than 70":
            ag = "70+"
        g_str = str(row.gender).strip()
        r_str = str(row.race).strip()
        if ag not in AGE_GROUP_MIDPOINT or g_str not in ("Male", "Female") or r_str not in ASIAN_RACES:
            continue
        records.append({
            "file": os.path.basename(full_p),
            "image_path": full_p,
            "age": float(AGE_GROUP_MIDPOINT[ag]),
            "age_group": ag,
            "gender": g_str,
            "gender_label": 0 if g_str == "Female" else 1,
            "race": r_str,
            "source": "FairFace",
            "identity_id": "",
        })
    return pd.DataFrame(records)


def _check_and_hash(path: str):
    try:
        if os.path.getsize(path) <= 0:
            return None
        with Image.open(path) as im:
            im.verify()
        h = hashlib.md5()
        with open(path, "rb") as f:
            h.update(f.read())
        return h.hexdigest()
    except Exception:
        return None


def validate_and_deduplicate(df: pd.DataFrame, max_workers: int = 8) -> pd.DataFrame:
    paths = df["image_path"].tolist()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        md5s = list(tqdm(ex.map(_check_and_hash, paths), total=len(paths), desc="Verify & MD5"))
    df = df.copy()
    df["md5"] = md5s
    n_bad = df["md5"].isna().sum()
    df = df.dropna(subset=["md5"]).reset_index(drop=True)
    n_before_dedup = len(df)
    df = df.drop_duplicates(subset=["md5"], keep="first").reset_index(drop=True)
    n_dup = n_before_dedup - len(df)
    print(f"[Quality Check] Loại {n_bad:,} ảnh hỏng | Loại {n_dup:,} ảnh trùng MD5 | Còn lại: {len(df):,} ảnh sạch")
    return df


def apply_flexible_cap(df: pd.DataFrame, cap: int = 2000, seed: int = 42) -> pd.DataFrame:
    sampled_parts = []
    for (ag, gl), grp in df.groupby(["age_group", "gender_label"], sort=False):
        if len(grp) > cap:
            sampled_parts.append(grp.sample(n=cap, random_state=seed))
        else:
            sampled_parts.append(grp)
    out = pd.concat(sampled_parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def split_leak_free(df: pd.DataFrame, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42):
    df = df.copy()
    # Gán group_key: dùng identity_id nếu có (AAF), ngược lại dùng md5 (unique per image)
    id_clean = df["identity_id"].fillna("").astype(str).str.strip()
    df["group_key"] = np.where((id_clean != "") & (id_clean.str.lower() != "nan"), id_clean, df["md5"])
    df["strat_col"] = df["age_group"].astype(str) + "_" + df["gender_label"].astype(str)

    # Lấy 1 đại diện cho mỗi group_key để chia group-aware + stratified
    grp_df = df.groupby("group_key", as_index=False).first()[["group_key", "strat_col"]]
    # Nếu có strat_col nào chỉ có < 3 group thì gộp vào nhãn chung để train_test_split không lỗi
    vc = grp_df["strat_col"].value_counts()
    rare = set(vc[vc < 3].index)
    grp_df["strat_safe"] = grp_df["strat_col"].where(~grp_df["strat_col"].isin(rare), "RARE")

    tr_keys, tmp_keys = train_test_split(
        grp_df["group_key"],
        test_size=(val_ratio + test_ratio),
        stratify=grp_df["strat_safe"],
        random_state=seed,
    )
    tmp_sub = grp_df[grp_df["group_key"].isin(tmp_keys)]
    vc2 = tmp_sub["strat_safe"].value_counts()
    rare2 = set(vc2[vc2 < 2].index)
    tmp_strat = tmp_sub["strat_safe"].where(~tmp_sub["strat_safe"].isin(rare2), "RARE")

    rel_test = test_ratio / (val_ratio + test_ratio)
    va_keys, te_keys = train_test_split(
        tmp_sub["group_key"],
        test_size=rel_test,
        stratify=tmp_strat,
        random_state=seed,
    )

    tr_set, va_set, te_set = set(tr_keys), set(va_keys), set(te_keys)
    train_df = df[df["group_key"].isin(tr_set)].reset_index(drop=True)
    val_df = df[df["group_key"].isin(va_set)].reset_index(drop=True)
    test_df = df[df["group_key"].isin(te_set)].reset_index(drop=True)

    # Kiểm tra rò rỉ dữ liệu bắt buộc = 0
    assert len(set(train_df["image_path"]) & set(val_df["image_path"])) == 0
    assert len(set(train_df["image_path"]) & set(test_df["image_path"])) == 0
    assert len(set(val_df["image_path"]) & set(test_df["image_path"])) == 0
    assert len(set(train_df["md5"]) & set(val_df["md5"])) == 0
    assert len(set(train_df["md5"]) & set(test_df["md5"])) == 0
    assert len(set(val_df["md5"]) & set(test_df["md5"])) == 0

    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        d["gender_str"] = d["gender"]
        d["gender"] = d["gender_label"]  # giữ cột gender = 0/1 tương thích AFADDataset

    return train_df, val_df, test_df


def _rgb_stats_worker(args):
    path, img_size = args
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB").resize((img_size, img_size)), dtype=np.float64) / 255.0
    s1 = arr.sum(axis=(0, 1))
    s2 = (arr ** 2).sum(axis=(0, 1))
    return s1, s2, img_size * img_size


def compute_train_norm_stats(train_df: pd.DataFrame, img_size: int = 224, max_workers: int = 8):
    paths = train_df["image_path"].tolist()
    total_s1 = np.zeros(3, dtype=np.float64)
    total_s2 = np.zeros(3, dtype=np.float64)
    total_px = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for s1, s2, npx in tqdm(
            ex.map(_rgb_stats_worker, [(p, img_size) for p in paths]),
            total=len(paths),
            desc=f"Compute Train Mean/Std ({img_size}x{img_size})",
        ):
            total_s1 += s1
            total_s2 += s2
            total_px += npx
    mean = total_s1 / total_px
    std = np.sqrt(np.maximum(total_s2 / total_px - mean ** 2, 1e-12))
    return {
        "img_size": img_size,
        "num_train_images": len(paths),
        "mean": [round(float(x), 6) for x in mean],
        "std": [round(float(x), 6) for x in std],
    }


def make_relative_paths(df: pd.DataFrame, project_root: str) -> pd.DataFrame:
    df = df.copy()
    root_abs = os.path.abspath(project_root).replace("\\", "/")
    df["image_path"] = df["image_path"].apply(
        lambda p: os.path.relpath(p, root_abs).replace("\\", "/")
    )
    return df


def main():
    p = argparse.ArgumentParser(description="Prepare, merge, cap (2000), split, and compute stats")
    p.add_argument("--project-root", type=str, default=".")
    p.add_argument("--afad-dir", type=str, default="data/AFAD-Full")
    p.add_argument("--utk-dir", type=str, default="data/raw/UTKFace")
    p.add_argument("--aaf-dir", type=str, default="data/raw/AAF")
    p.add_argument("--fairface-dir", type=str, default="data/raw/FairFace")
    p.add_argument("--processed-dir", type=str, default="data/processed")
    p.add_argument("--splits-dir", type=str, default="data/splits")
    p.add_argument("--cap", type=int, default=2000)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.processed_dir, exist_ok=True)
    os.makedirs(args.splits_dir, exist_ok=True)

    dfs = []
    for name, fn, dpath in [
        ("AFAD-Full", parse_afad, args.afad_dir),
        ("UTKFace", parse_utkface, args.utk_dir),
        ("AAF", parse_aaf, args.aaf_dir),
        ("FairFace", parse_fairface, args.fairface_dir),
    ]:
        part = fn(dpath)
        print(f"[Parse {name}] Thu thập được: {len(part):,} bản ghi thô")
        if len(part) > 0:
            dfs.append(part)

    merged = pd.concat(dfs, ignore_index=True)
    print(f"\n[Merge] Tổng bản ghi trước kiểm tra chất lượng: {len(merged):,}")
    print(merged["source"].value_counts().to_dict())

    merged_clean = validate_and_deduplicate(merged)
    merged_clean_rel = make_relative_paths(merged_clean, args.project_root)
    merged_clean_rel.to_csv(os.path.join(args.processed_dir, "metadata_merged_raw.csv"), index=False)

    print("\n[Trước Flexible Cap] Phân bố Age Group x Gender:")
    ct_before = pd.crosstab(merged_clean["age_group"], merged_clean["gender"], margins=True).reindex(AGE_GROUPS_9 + ["All"])
    print(ct_before)

    balanced = apply_flexible_cap(merged_clean, cap=args.cap, seed=args.seed)
    balanced_rel = make_relative_paths(balanced, args.project_root)
    balanced_rel.to_csv(os.path.join(args.processed_dir, "metadata_balanced.csv"), index=False)

    print(f"\n[Sau Flexible Cap = {args.cap}] Tổng mẫu cân bằng: {len(balanced):,}")
    print("Nguồn sau cân bằng:", balanced["source"].value_counts().to_dict())
    ct_after = pd.crosstab(balanced["age_group"], balanced["gender"], margins=True).reindex(AGE_GROUPS_9 + ["All"])
    print(ct_after)

    train_df, val_df, test_df = split_leak_free(balanced, val_ratio=0.15, test_ratio=0.15, seed=args.seed)

    # Lưu splits với đường dẫn tương đối để chạy được cả trên Windows lẫn Colab
    make_relative_paths(train_df, args.project_root).to_csv(os.path.join(args.splits_dir, "train.csv"), index=False)
    make_relative_paths(val_df, args.project_root).to_csv(os.path.join(args.splits_dir, "val.csv"), index=False)
    make_relative_paths(test_df, args.project_root).to_csv(os.path.join(args.splits_dir, "test.csv"), index=False)

    print(f"\n[Splits 70/15/15 - Leak-free] Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # Tính Mean & Std trên tập Train (224x224)
    norm_stats = compute_train_norm_stats(train_df, img_size=args.img_size)
    with open(os.path.join(args.processed_dir, "norm_stats.json"), "w", encoding="utf-8") as f:
        json.dump(norm_stats, f, indent=2)
    print("\n[Norm Stats trên Train 224x224]:", norm_stats)

    # Tính & lưu công thức chuẩn hóa nhãn Age
    age_mean = float(train_df["age"].mean())
    age_std = float(train_df["age"].std())
    age_min = float(train_df["age"].min())
    age_max = float(train_df["age"].max())
    age_scaler = {
        "seed": args.seed,
        "train_age_mean": round(age_mean, 4),
        "train_age_std": round(age_std, 4),
        "train_age_min": age_min,
        "train_age_max": age_max,
        "zscore_forward": "(age - train_age_mean) / train_age_std",
        "zscore_inverse": "age_norm * train_age_std + train_age_mean",
        "minmax_100_forward": "age / 100.0",
        "minmax_100_inverse": "age_norm * 100.0",
    }
    with open(os.path.join(args.processed_dir, "age_scaler.json"), "w", encoding="utf-8") as f:
        json.dump(age_scaler, f, indent=2)
    print("[Age Scaler]:", age_scaler)


if __name__ == "__main__":
    main()
