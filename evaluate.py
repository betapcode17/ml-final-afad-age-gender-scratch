# Entry point chạy quá trình đánh giá (Evaluation) trên tập Test
# Hỗ trợ truyền --model <tên_mô_hình> để đánh giá bất kỳ mô hình nào trong MODEL_REGISTRY.

import argparse
from src.evaluation.evaluate import run_evaluation
from src.models.builder import get_available_models


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate AFAD Multi-task CNN")
    p.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=get_available_models(),
        help="Tên mô hình đã đăng ký trong src/models/builder.py",
    )
    p.add_argument("--splits-dir", type=str, default="dataset/metadata")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--img-size", type=int, default=224)
    return p.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())
