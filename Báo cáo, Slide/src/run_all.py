#!/usr/bin/env python
"""Điểm vào duy nhất: sinh dữ liệu, chạy thí nghiệm A/B/C, vẽ hình.

    python run_all.py --exp all                 # chạy toàn bộ bằng PyTorch (khuyến nghị)
    python run_all.py --exp data                # chỉ sinh + khảo sát dữ liệu
    python run_all.py --exp A  / B / C          # từng nhóm (A2, B2: đối chứng)
    python run_all.py --exp final               # hình y-ŷ, bề mặt xấp xỉ (cần đã chạy B, C)
    python run_all.py --engine numpy --exp all  # engine tham chiếu NumPy (không cần PyTorch)
    python run_all.py --quick --exp all         # chạy thử nhanh (30 epoch, 1 seed)
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src import data as D, experiments as E, plotting as P  # noqa: E402


def get_engine(name):
    if name in ("torch", "auto"):
        try:
            import torch  # noqa: F401
            from src import engine_torch
            return engine_torch
        except ImportError:
            if name == "torch":
                raise SystemExit("Chưa cài PyTorch: pip install torch (hoặc dùng --engine numpy)")
            print("[CẢNH BÁO] Không có PyTorch -> dùng engine NumPy tham chiếu.")
    from src import engine_numpy
    return engine_numpy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all", choices=["all", "data", "A", "A2", "B", "B2", "C", "final"])
    ap.add_argument("--engine", default="auto", choices=["auto", "torch", "numpy"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(ROOT))
    a = ap.parse_args()
    out = Path(a.out)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    seeds = [0] if a.quick else list(range(a.seeds))

    splits = D.load_splits(out / "data")
    todo = ["data", "A", "A2", "B", "B2", "C", "final"] if a.exp == "all" else [a.exp]
    res = out / "results"
    res.mkdir(exist_ok=True)

    if "data" in todo:
        D.describe(splits).to_csv(res / "data_stats.csv", index=False)
        P.fig_data(splits, fig_dir)
        print("Đã sinh dữ liệu + hình khảo sát.")
    engine = None
    if any(t in todo for t in ["A", "A2", "B", "B2", "C"]):
        engine = get_engine(a.engine)
        (res / "engine.txt").write_text(engine.NAME)
        print("Engine:", engine.NAME)
    for exp in [t for t in todo if t in ("A", "A2", "B", "B2", "C")]:
        E.run_group(engine, exp, splits, out, seeds, a.device, a.quick)

    if "final" in todo or a.exp == "all":
        make_figures(out, seeds, splits)


def make_figures(out, seeds, splits):
    res, fig = out / "results", out / "figures"
    S = {e: pd.read_csv(res / f"summary_{e}.csv") for e in ["A", "A2", "B", "B2", "C"] if (res / f"summary_{e}.csv").exists()}
    logs = res / "logs"
    if "A" in S:
        P.fig_sweep(S["A"], "A", "Số mẫu huấn luyện A", fig / "fig04_expA.png", "Nhóm A")
    if "A" in S and "A2" in S:
        P.fig_A_compare(S["A"], S["A2"], fig / "fig05_expA_control.png")
    if "B" in S:
        P.fig_sweep(S["B"], "B", "Số nơ-ron tầng ẩn B (thang log)", fig / "fig06_expB.png", "Nhóm B (A=2000)", logx=True)
    if "B2" in S:
        P.fig_sweep(S["B2"], "B", "Số nơ-ron tầng ẩn B (thang log)", fig / "fig07_expB2.png", "Đối chứng B2 (A=200)", logx=True)
    if "C" in S:
        P.fig_C(S["C"], logs, seeds, fig / "fig08_expC.png")
    if (logs / "C_C1_s0.csv").exists() and (logs / "A_A200_s0.csv").exists():
        P.fig_curves(logs, fig / "fig09_loss_curves.png",
                     [("A", "A200", "A=200, B=10 (1000 epoch)"), ("A2", "A200", "A=200, B=10 (đối chứng 10000 epoch)"),
                      ("C", "C1", "A=2000, B=10, C=1 (baseline)")])
    # chọn mô hình cuối: val MSE trung bình thấp nhất trong các cấu hình A=2000 (nhóm B, C) -> seed có val thấp nhất
    cand = pd.concat([S[e] for e in ["B", "C"] if e in S], ignore_index=True)
    best = cand.loc[cand.val_mse_mean.idxmin()]
    exp = best.exp
    runs = pd.read_csv(res / f"runs_{exp}.csv")
    runs = runs[runs.tag == best.tag]
    r = runs.loc[runs.val_mse.idxmin()]
    st = load_state(out / "models" / f"{exp}_{best.tag}_s{int(r.seed)}")
    P_metrics = {"mse": r.test_mse, "mae": r.test_mae}
    extra = P.fig_final(st, "tanh", splits, P_metrics, fig, f"{exp}:{best.tag}")
    info = dict(model=f"{exp}:{best.tag}", seed=int(r.seed), hidden=r.hidden, val_mse=float(r.val_mse),
                test_mse=float(r.test_mse), test_mae=float(r.test_mae), test_rmse=float(r.test_rmse), **extra)
    (res / "final_model.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print("Mô hình cuối:", info)


def load_state(base):
    import numpy as np
    if Path(str(base) + ".npz").exists():
        z = np.load(str(base) + ".npz")
        return {k: z[k] for k in z.files}
    import torch
    ck = torch.load(str(base) + ".pt", map_location="cpu", weights_only=False)
    sd, st, i = ck["state_dict"], {}, 0
    keys = [k for k in sd if k.endswith("weight")]
    for k in keys:
        st[f"W{i}"] = sd[k].numpy().astype("float64")
        st[f"b{i}"] = sd[k.replace("weight", "bias")].numpy().astype("float64")
        i += 1
    return st


if __name__ == "__main__":
    main()
