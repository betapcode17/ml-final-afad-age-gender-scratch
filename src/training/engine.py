# Các hàm chạy từng epoch: train_one_epoch() và validate()
# Yêu cầu:
# - train_one_epoch: forward -> tính Multi-task loss -> backward -> optimizer.step()
# - validate: torch.no_grad() -> tính loss, Gender Accuracy/F1/AUC, Age MAE/RMSE trên tập Val
# - Trả về dictionary các chỉ số của epoch hiện tại
