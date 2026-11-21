# Logic điều phối quá trình huấn luyện (Training Pipeline)
# Hỗ trợ WeightedRandomSampler, Kiểu A (multiclass) và Kiểu B (coral) trên 30 epochs.

import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
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
    age_mode = getattr(args, "age_mode", "coral")

    train_tf, val_tf = get_transforms(args.img_size, args.augment, args.vflip)
    train_ds = AFADDataset(os.path.join(args.splits_dir, "train.csv"), train_tf)
    val_ds = AFADDataset(os.path.join(args.splits_dir, "val.csv"), val_tf)

    strat_key = train_ds.df["age_group"].astype(str) + "_" + train_ds.df["gender"].astype(str)
    counts = strat_key.value_counts().to_dict()
    sample_weights = strat_key.map(lambda k: 1.0 / counts[k]).values
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
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

    model = build_model(args.model, age_mode=age_mode).to(device)
    criterion = MultiTaskLoss(age_mode=age_mode, w_gender=args.w_gender, w_age=args.w_age)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch, best_val_loss = 0, float("inf")
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(args.resume, model, optimizer, device)

    tag = f"{args.model}_{age_mode}"
    history = []
    for epoch in range(start_epoch + 1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va = validate(model, val_loader, criterion, device)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "tr_loss": tr["loss"],
            "tr_gender_loss": tr["gender_loss"],
            "tr_age_loss": tr["age_loss"],
            "tr_acc": tr["gender_acc"],
            "tr_age_grp_acc": tr["age_acc"],
            "tr_age_adj_acc": tr["age_adj_acc"],
            "val_loss": va["loss"],
            "val_gender_loss": va["gender_loss"],
            "val_age_loss": va["age_loss"],
            "val_acc": va["gender_acc"],
            "val_age_grp_acc": va["age_acc"],
            "val_age_adj_acc": va["age_adj_acc"],
        })
        print(
            f"[{tag}] Epoch {epoch:02d}/{args.epochs} | "
            f"Tr Loss {tr['loss']:.4f} G_Acc {tr['gender_acc']*100:.1f}% A_Acc {tr['age_acc']*100:.1f}% | "
            f"Val Loss {va['loss']:.4f} G_Acc {va['gender_acc']*100:.1f}% A_Acc {va['age_acc']*100:.1f}%"
        )
        if va["loss"] < best_val_loss:
            best_val_loss = va["loss"]
            save_checkpoint(
                {
                    "epoch": epoch,
                    "age_mode": age_mode,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                os.path.join(args.save_dir, f"{tag}_best.pth"),
            )

    save_checkpoint(
        {
            "epoch": args.epochs,
            "age_mode": age_mode,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        },
        os.path.join(args.save_dir, f"{tag}_epoch_{args.epochs}.pth"),
    )
    save_history(history, os.path.join("results", tag, "history.csv"))
