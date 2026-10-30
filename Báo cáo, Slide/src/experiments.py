"""Chạy nhóm thí nghiệm A/B/C (+ đối chứng A2, B2), ghi log từng lần chạy và bảng tổng hợp.

Baseline (chọn lr bằng validation, xem README): tanh, Adam lr=3e-3, batch 64, tối đa 1000 epoch,
dừng sớm patience=100 trên val loss, checkpoint theo val loss thấp nhất. Seed: 0..4 (khởi tạo + thứ tự batch).
Dữ liệu cố định (DATA_SEED=2026): chỉ đổi yếu tố đang khảo sát.
"""
from pathlib import Path
import pandas as pd
from . import data as D

BASE = dict(activation="tanh", lr=3e-3, batch_size=64, epochs=1000, patience=100)
B_LIST = list(range(1, 11)) + [16, 32, 64]     # B = 1..10 mở rộng thêm 16, 32, 64
A_LIST = list(range(200, 2001, 200))
C_LIST = [1, 2, 3, 4, 5]
W_BASE = 10                                     # số nơ-ron mỗi tầng ẩn ở nhóm C và A


def _plans():
    """exp -> list(tag, A, hidden, epochs)."""
    E = BASE["epochs"]
    scale = lambda A: int(E * 2000 / A)         # cân bằng số bước cập nhật theo A
    return {
        "A":  [(f"A{A}", A, (W_BASE,), E) for A in A_LIST],
        "A2": [(f"A{A}", A, (W_BASE,), scale(A)) for A in A_LIST],
        "B":  [(f"B{B}", 2000, (B,), E) for B in B_LIST],
        "B2": [(f"B{B}", 200, (B,), scale(200)) for B in B_LIST],
        "C":  [(f"C{C}", 2000, (W_BASE,) * C, E) for C in C_LIST],
    }


def run_group(engine, exp, splits, out_dir, seeds, device="cpu", quick=False, save_models=True, verbose=True):
    out_dir = Path(out_dir)
    (out_dir / "results" / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    Xva, yva = D.to_arrays(splits["val"])
    Xte, yte = D.to_arrays(splits["test"])
    rows = []
    for tag, A, hidden, epochs in _plans()[exp]:
        Xtr, ytr = D.to_arrays(splits["train"], A)
        if quick:
            epochs = min(epochs, 30)
        pat = max(BASE["patience"], epochs // 10)
        for seed in seeds:
            res = engine.train_run(Xtr, ytr, Xva, yva, Xte, yte, hidden, BASE["activation"], BASE["lr"],
                                   BASE["batch_size"], epochs, pat, seed, device)
            m = res["metrics"]
            row = dict(exp=exp, tag=tag, A=A, B=hidden[0] if len(hidden) == 1 else W_BASE, C=len(hidden),
                       hidden=str(list(hidden)), seed=seed, engine=engine.NAME, lr=BASE["lr"],
                       batch_size=BASE["batch_size"], epochs_max=epochs, patience=pat,
                       epochs_run=res["epochs_run"], best_epoch=res["best_epoch"],
                       train_time_s=round(res["train_time_s"], 4),
                       train_mse=m["train"]["mse"], val_mse=m["val"]["mse"], test_mse=m["test"]["mse"],
                       train_mae=m["train"]["mae"], val_mae=m["val"]["mae"], test_mae=m["test"]["mae"],
                       test_rmse=m["test"]["rmse"], params=sum(w.size for w in res["state"].values()))
            rows.append(row)
            pd.DataFrame(res["history"]).to_csv(out_dir / "results" / "logs" / f"{exp}_{tag}_s{seed}.csv", index=False)
            if save_models:
                engine.save_checkpoint(res, out_dir / "models" / f"{exp}_{tag}_s{seed}",
                                       dict(hidden=hidden, activation=BASE["activation"], seed=seed))
            # lưu state để dựng hình cuối (không ghi ra đĩa ở đây)
            row["_state"] = res["state"]
        if verbose:
            r = pd.DataFrame([{k: v for k, v in x.items() if k != "_state"} for x in rows if x["tag"] == tag])
            print(f"[{exp}] {tag:5s} A={A:4d} hidden={list(hidden)} val={r.val_mse.mean():.3e} "
                  f"test={r.test_mse.mean():.3e} time={r.train_time_s.mean():.1f}s", flush=True)
    df = pd.DataFrame([{k: v for k, v in x.items() if k != "_state"} for x in rows])
    df.to_csv(out_dir / "results" / f"runs_{exp}.csv", index=False)
    summarize(df).to_csv(out_dir / "results" / f"summary_{exp}.csv", index=False)
    return df, rows


def summarize(df):
    df = df.assign(ms_per_epoch=df.train_time_s / df.epochs_run * 1000.0)
    cols = ["ms_per_epoch", "train_mse", "val_mse", "test_mse", "train_mae", "val_mae", "test_mae", "test_rmse",
            "train_time_s", "best_epoch", "epochs_run", "params"]
    g = df.groupby(["exp", "tag", "A", "B", "C"], sort=False)[cols].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g["n_seeds"] = df.groupby(["exp", "tag", "A", "B", "C"], sort=False).size()
    return g.reset_index()
