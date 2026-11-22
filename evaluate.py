# Entry point chạy quá trình đánh giá (Evaluation) trên tập Test
# Hỗ trợ --model <resnet18|resnet18_se|resnet18_cbam_ms>, --age-mode <multiclass|coral|coral_v2|coral_v3> và --tta

import argparse
from src.evaluation.evaluate import run_evaluation
from src.models.builder import get_available_models


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Asian Multi-task CNN")
    p.add_argument(
        "--model",
        type=str,
        default="resnet18_hybrid",
        choices=get_available_models(),
        help="Tên mô hình đã đăng ký trong src/models/builder.py",
    )
    p.add_argument(
        "--age-mode",
        type=str,
        default="coral_v4",
        choices=["multiclass", "coral", "coral_v2", "coral_v3", "coral_v4"],
        help="multiclass (02a) | coral (02b) | coral_v2 (02c) | coral_v3 (02d) | coral_v4 (02e)",
    )
    p.add_argument("--splits-dir", type=str, default="dataset/metadata")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--tta", action="store_true", help="Bật Test-Time Augmentation (gốc + lật ngang)")
    return p.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())
