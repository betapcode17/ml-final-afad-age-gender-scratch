# Các hàm chạy từng epoch: train_one_epoch() và validate()
# Yêu cầu:
# - train_one_epoch: forward -> tính Multi-task loss -> backward -> optimizer.step()
# - validate: torch.no_grad() -> tính loss, Gender Accuracy/F1/AUC, Age MAE/RMSE trên tập Val
# - Trả về dictionary các chỉ số của epoch hiện tại
import torch
from tqdm import tqdm
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, mae_sum, n = 0.0, 0, 0.0, 0
    for imgs, genders, ages in tqdm(loader, desc="Train", leave=False):
        imgs, genders, ages = imgs.to(device), genders.to(device), ages.to(device)
        optimizer.zero_grad()
        g_logits, a_pred = model(imgs)
        loss = criterion(g_logits, a_pred, genders, ages)
        loss.backward()
        optimizer.step()
        bs = imgs.size(0)
        loss_sum += loss.item() * bs
        correct += (g_logits.argmax(1) == genders).sum().item()
        mae_sum += torch.abs(a_pred - ages).sum().item()
        n += bs
    return {"loss": loss_sum / n, "acc": correct / n, "mae": mae_sum / n}
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, mae_sum, n = 0.0, 0, 0.0, 0
    for imgs, genders, ages in tqdm(loader, desc="Val", leave=False):
        imgs, genders, ages = imgs.to(device), genders.to(device), ages.to(device)
        g_logits, a_pred = model(imgs)
        loss = criterion(g_logits, a_pred, genders, ages)
        bs = imgs.size(0)
        loss_sum += loss.item() * bs
        correct += (g_logits.argmax(1) == genders).sum().item()
        mae_sum += torch.abs(a_pred - ages).sum().item()
        n += bs
    return {"loss": loss_sum / n, "acc": correct / n, "mae": mae_sum / n}