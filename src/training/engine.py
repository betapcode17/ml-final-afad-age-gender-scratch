# Các hàm chạy từng epoch: train_one_epoch() và validate()
# Hỗ trợ Kiểu A (multiclass), Kiểu B (coral), Kiểu B+ (coral_v2) và Kiểu B++ (coral_v3)

import torch
from tqdm import tqdm


def _decode_age_group(age_logits, age_mode):
    if age_mode in ("coral", "coral_v2", "coral_v3"):
        return (torch.sigmoid(age_logits) > 0.5).sum(dim=1).long()
    return age_logits.argmax(dim=1).long()


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0):
    model.train()
    loss_sum, g_loss_sum, a_loss_sum = 0.0, 0.0, 0.0
    g_correct, a_correct, a_adj_correct, n = 0, 0, 0, 0
    for imgs, genders, age_groups in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device)
        genders = genders.to(device)
        age_groups = age_groups.to(device)
        optimizer.zero_grad()

        g_logit, a_logits = model(imgs)
        loss, lg, la = criterion(g_logit, a_logits, genders, age_groups, return_parts=True)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = imgs.size(0)
        loss_sum += loss.item() * bs
        g_loss_sum += lg.item() * bs
        a_loss_sum += la.item() * bs

        g_pred = (torch.sigmoid(g_logit) >= 0.5).long()
        a_pred = _decode_age_group(a_logits, model.age_mode)
        g_correct += (g_pred == genders.long()).sum().item()
        a_correct += (a_pred == age_groups).sum().item()
        a_adj_correct += (torch.abs(a_pred - age_groups) <= 1).sum().item()
        n += bs

    return {
        "loss": loss_sum / n,
        "gender_loss": g_loss_sum / n,
        "age_loss": a_loss_sum / n,
        "gender_acc": g_correct / n,
        "age_acc": a_correct / n,
        "age_adj_acc": a_adj_correct / n,
    }


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_sum, g_loss_sum, a_loss_sum = 0.0, 0.0, 0.0
    g_correct, a_correct, a_adj_correct, n = 0, 0, 0, 0
    for imgs, genders, age_groups in tqdm(loader, desc="Val", leave=False):
        imgs = imgs.to(device)
        genders = genders.to(device)
        age_groups = age_groups.to(device)

        g_logit, a_logits = model(imgs)
        loss, lg, la = criterion(g_logit, a_logits, genders, age_groups, return_parts=True)

        bs = imgs.size(0)
        loss_sum += loss.item() * bs
        g_loss_sum += lg.item() * bs
        a_loss_sum += la.item() * bs

        g_pred = (torch.sigmoid(g_logit) >= 0.5).long()
        a_pred = _decode_age_group(a_logits, model.age_mode)
        g_correct += (g_pred == genders.long()).sum().item()
        a_correct += (a_pred == age_groups).sum().item()
        a_adj_correct += (torch.abs(a_pred - age_groups) <= 1).sum().item()
        n += bs

    return {
        "loss": loss_sum / n,
        "gender_loss": g_loss_sum / n,
        "age_loss": a_loss_sum / n,
        "gender_acc": g_correct / n,
        "age_acc": a_correct / n,
        "age_adj_acc": a_adj_correct / n,
    }
