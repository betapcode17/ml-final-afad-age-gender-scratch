# Logic điều phối quá trình huấn luyện (Training Pipeline)
# Tự động nhận diện mô hình qua args.model và lưu checkpoint + history theo đúng tên mô hình.

import os
import torch
from torch.utils.data import DataLoader
from src.datasets.afad_dataset import AFADDataset
from src.datasets.transforms import get_transforms
from src.models.builder import build_model
from src.training.engine import train_one_epoch, validate
from src.training.losses import MultiTaskLoss
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.logger import save_history
from src.utils.seed import set_seed


def run_training(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf, val_tf = get_transforms(args.img_size, args.augment, args.vflip)
    train_ds = AFADDataset(os.path.join(args.splits_dir, "train.csv"), train_tf)
    val_ds = AFADDataset(os.path.join(args.splits_dir, "val.csv"), val_tf)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(args.model).to(device)
    criterion = MultiTaskLoss(w_gender=args.w_gender, w_age=args.w_age)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch, best_val_loss = 0, float("inf")
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(args.resume, model, optimizer, device)

    history = []
    for epoch in range(start_epoch + 1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va = validate(model, val_loader, criterion, device)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "tr_loss": tr["loss"],
            "tr_acc": tr["acc"],
            "tr_mae": tr["mae"],
            "val_loss": va["loss"],
            "val_acc": va["acc"],
            "val_mae": va["mae"],
        })
        print(
            f"[{args.model}] Epoch {epoch:02d}/{args.epochs} | "
            f"Tr Loss {tr['loss']:.4f} Acc {tr['acc']:.4f} MAE {tr['mae']:.2f} | "
            f"Val Loss {va['loss']:.4f} Acc {va['acc']:.4f} MAE {va['mae']:.2f}"
        )
        if va["loss"] < best_val_loss:
            best_val_loss = va["loss"]
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                os.path.join(args.save_dir, f"{args.model}_best.pth"),
            )

    save_checkpoint(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        },
        os.path.join(args.save_dir, f"{args.model}_epoch_{args.epochs}.pth"),
    )
    save_history(history, os.path.join("results", args.model, "history.csv"))
