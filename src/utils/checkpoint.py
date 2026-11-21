import os
import torch

def save_checkpoint(state,filepath): 
    os.makedirs(os.path.dirname(filepath),exist_ok=True)
    torch.save(state,filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None, device="cpu"):
    ckpt = torch.load(filepath, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("best_val_loss", float("inf"))