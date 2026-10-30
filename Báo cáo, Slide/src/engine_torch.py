"""Engine chính: huấn luyện bằng PyTorch (nn.Module, DataLoader, MSELoss, Adam)."""
import copy
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .common import regression_metrics
from .model import MLP

NAME = "pytorch"


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, X, device="cpu"):
    model.eval()
    return model(torch.as_tensor(X, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)


@torch.no_grad()
def eval_loss(model, X, y, criterion):
    model.eval()
    return criterion(model(X), y).item()


def state_to_numpy(model):
    """nn.Sequential: các nn.Linear lần lượt -> W0,b0,W1,b1,..."""
    st, i = {}, 0
    for m in model.net:
        if isinstance(m, nn.Linear):
            st[f"W{i}"] = m.weight.detach().cpu().numpy().astype(np.float64)
            st[f"b{i}"] = m.bias.detach().cpu().numpy().astype(np.float64)
            i += 1
    return st


def train_run(Xtr, ytr, Xva, yva, Xte, yte, hidden, activation="tanh", lr=1e-2,
              batch_size=64, epochs=300, patience=60, seed=0, device="cpu"):
    set_seed(seed)
    dev = torch.device(device)
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)
    Xtr_t, ytr_t = t(Xtr), t(ytr).unsqueeze(1)
    Xva_t, yva_t = t(Xva), t(yva).unsqueeze(1)

    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch_size, shuffle=True, generator=g)

    model = MLP(in_dim=Xtr.shape[1], hidden=tuple(hidden), activation=activation).to(dev)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history, best_val, best_epoch, best_state, bad = [], float("inf"), 0, None, 0
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:               # vòng lặp huấn luyện tự viết
            optimizer.zero_grad()           # 1) xoá gradient cũ
            loss = criterion(model(xb), yb)  # 2) forward + loss
            loss.backward()                 # 3) backpropagation
            optimizer.step()                # 4) cập nhật tham số
        train_loss = eval_loss(model, Xtr_t, ytr_t, criterion)   # model.eval() + no_grad
        val_loss = eval_loss(model, Xva_t, yva_t, criterion)
        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss))
        if val_loss < best_val - 1e-12:     # checkpoint theo validation loss
            best_val, best_epoch, bad = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    train_time = time.perf_counter() - t0

    model.load_state_dict(best_state)       # khôi phục checkpoint tốt nhất
    metrics = {k: regression_metrics(y, predict(model, X, device))
               for k, (X, y) in {"train": (Xtr, ytr), "val": (Xva, yva), "test": (Xte, yte)}.items()}
    return dict(history=history, best_epoch=best_epoch, epochs_run=len(history), train_time_s=train_time,
                state=state_to_numpy(model), metrics=metrics, torch_state=best_state)


def save_checkpoint(res, path, meta):
    torch.save({"state_dict": res["torch_state"], "meta": meta}, str(path) + ".pt")
