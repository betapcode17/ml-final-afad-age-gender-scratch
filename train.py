# Entry point chạy quá trình huấn luyện (Training)
# Hỗ trợ truyền --model <tên_mô_hình> để dùng chung template cho mọi thành viên trong nhóm.

import argparse
import yaml
from src.models.builder import get_available_models
from src.training.train import run_training


def parse_args():
    p = argparse.ArgumentParser(description="Train AFAD Multi-task CNN from scratch")
    p.add_argument("--config", type=str, default="", help="Đường dẫn file YAML (tùy chọn)")
    p.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=get_available_models(),
        help="Tên mô hình đã đăng ký trong src/models/builder.py",
    )
    p.add_argument("--splits-dir", type=str, default="dataset/metadata")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--vflip", action="store_true")
    p.add_argument("--w-gender", type=float, default=1.0)
    p.add_argument("--w-age", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default="")
    args = p.parse_args()
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            for k, v in cfg.items():
                if hasattr(args, k):
                    setattr(args, k, v)
    return args


if __name__ == "__main__":
    run_training(parse_args())
