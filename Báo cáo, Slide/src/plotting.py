"""Vẽ hình cho báo cáo (matplotlib, backend Agg)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from . import data as D
from .common import numpy_forward

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 160, "axes.grid": True, "grid.alpha": 0.3,
                     "font.size": 10, "axes.titlesize": 11})
COL = {"train": "#2a9d8f", "val": "#e9a23b", "test": "#d1495b"}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _grid(n=70):
    g1 = np.linspace(*D.X1_RANGE, n)
    g2 = np.linspace(*D.X2_RANGE, n)
    G1, G2 = np.meshgrid(g1, g2)
    return G1, G2, np.stack([G1.ravel(), G2.ravel()], 1)


def fig_data(splits, fig_dir):
    fig = plt.figure(figsize=(14, 4.4))
    for i, (k, c, t) in enumerate([("train", COL["train"], "Train (pool 2000 mẫu)"),
                                   ("val", COL["val"], "Validation (200 mẫu)"),
                                   ("test", COL["test"], "Test (100 mẫu)")]):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        df = splits[k]
        ax.scatter(df.x1, df.x2, df.y, s=6 if k == "train" else 14, c=c, alpha=0.7)
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_zlabel("$y=f(x)$"); ax.set_title(t)
        ax.view_init(25, -55)
    _save(fig, Path(fig_dir) / "fig01_data_scatter3d.png")

    G1, G2, P = _grid()
    Z = D.gaussian_pdf(P).reshape(G1.shape)
    fig = plt.figure(figsize=(11, 4.6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(G1, G2, Z, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_zlabel("$f(x)$"); ax.set_title("Bề mặt hàm Gaussian thật")
    ax.view_init(28, -55)
    ax = fig.add_subplot(1, 2, 2)
    cs = ax.contourf(G1, G2, Z, 20, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="$f(x)$")
    tr = splits["train"]
    ax.scatter(tr.x1[:500], tr.x2[:500], s=3, c="white", alpha=0.6, label="500 mẫu train đầu")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_title("Đường đồng mức và mẫu train"); ax.legend(loc="upper right")
    _save(fig, Path(fig_dir) / "fig02_true_surface.png")

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, c, t in zip(axs, ["x1", "x2", "y"], ["$x_1$", "$x_2$", "$y$"]):
        for k in ["train", "val", "test"]:
            ax.hist(splits[k][c], bins=30, alpha=0.45, density=True, color=COL[k], label=k)
        ax.set_xlabel(t); ax.set_ylabel("mật độ"); ax.set_title(f"Phân bố {t}")
        if c == "y":
            ax.set_yscale("log")
    axs[0].legend()
    _save(fig, Path(fig_dir) / "fig03_distributions.png")


def _band(ax, df, x, col, label, color, marker="o"):
    m, s = df[f"{col}_mean"].to_numpy(), df[f"{col}_std"].fillna(0).to_numpy()
    lo = np.minimum(s, 0.8 * m)          # tránh thanh sai số xuống ≤ 0 trên thang log (cắt tại 20% giá trị TB)
    ax.errorbar(x, m, yerr=[lo, s], marker=marker, ms=4, capsize=3, color=color, label=label)


def fig_sweep(summ, xcol, xlabel, path, title, logx=False, mae=True):
    x = summ[xcol].to_numpy()
    ncol = 2 if mae else 1
    fig, axs = plt.subplots(1, ncol, figsize=(11 if mae else 6, 4))
    axs = np.atleast_1d(axs)
    for k in ["train", "val", "test"]:
        _band(axs[0], summ, x, f"{k}_mse", k, COL[k])
    axs[0].set_yscale("log"); axs[0].set_xlabel(xlabel); axs[0].set_ylabel("MSE (thang log)")
    axs[0].set_title(f"{title}: MSE (trung bình ± độ lệch chuẩn)"); axs[0].legend()
    if mae:
        for k in ["train", "val", "test"]:
            _band(axs[1], summ, x, f"{k}_mae", k, COL[k])
        axs[1].set_yscale("log"); axs[1].set_xlabel(xlabel); axs[1].set_ylabel("MAE (thang log)")
        axs[1].set_title(f"{title}: MAE"); axs[1].legend()
    if logx:
        for a in axs:
            a.set_xscale("log")
    _save(fig, path)


def fig_A_compare(sA, sA2, path):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for s, lab, c in [(sA, "Cố định 1000 epoch", "#264653"), (sA2, "Cân bằng số bước cập nhật", "#e76f51")]:
        _band(axs[0], s, s.A, "test_mse", lab, c)
        gap = s.copy()
        axs[1].plot(s.A, s.test_mse_mean / s.train_mse_mean, marker="o", color=c, label=lab)
    axs[0].set_yscale("log"); axs[0].set_xlabel("Số mẫu huấn luyện A"); axs[0].set_ylabel("Test MSE")
    axs[0].set_title("Test MSE theo A: hai cách đặt ngân sách huấn luyện"); axs[0].legend()
    axs[1].set_xlabel("Số mẫu huấn luyện A"); axs[1].set_ylabel("Test MSE / Train MSE")
    axs[1].set_title("Tỉ số test/train (khoảng cách tổng quát hoá)"); axs[1].axhline(1, color="gray", ls="--"); axs[1].legend()
    _save(fig, path)


def _mean_curve(logs_dir, exp, tag, seeds, col):
    curves = [pd.read_csv(Path(logs_dir) / f"{exp}_{tag}_s{s}.csv")[col].to_numpy() for s in seeds]
    L = min(len(c) for c in curves)
    return np.mean([c[:L] for c in curves], 0)


def fig_C(sC, logs_dir, seeds, path):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    x = sC.C.to_numpy()
    for k in ["train", "val", "test"]:
        _band(axs[0], sC, x, f"{k}_mse", k, COL[k])
    axs[0].set_yscale("log"); axs[0].set_xlabel("Số tầng ẩn C (10 nơ-ron/tầng)"); axs[0].set_ylabel("MSE (thang log)")
    axs[0].set_title("MSE theo độ sâu"); axs[0].legend(); axs[0].set_xticks(x)
    cm = plt.cm.viridis(np.linspace(0, 0.9, len(sC)))
    for c, (_, r) in zip(cm, sC.iterrows()):
        axs[1].plot(_mean_curve(logs_dir, "C", r.tag, seeds, "val_loss"), color=c, label=f"C={int(r.C)}")
    axs[1].set_yscale("log"); axs[1].set_xscale("log"); axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("Validation loss (MSE)")
    axs[1].set_title("Hội tụ của validation loss (TB các seed)"); axs[1].legend()
    _band(axs[2], sC, x, "train_time_s", "tổng thời gian (s)", "#6a4c93")
    axs[2].set_xlabel("Số tầng ẩn C"); axs[2].set_ylabel("Tổng thời gian huấn luyện (giây)"); axs[2].set_xticks(x)
    ax2 = axs[2].twinx(); ax2.grid(False)
    _band(ax2, sC, x, "ms_per_epoch", "ms / epoch", "#2a9d8f", marker="s")
    ax2.set_ylabel("Thời gian mỗi epoch (ms)")
    h1, l1 = axs[2].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    axs[2].legend(h1 + h2, l1 + l2, loc="upper left")
    axs[2].set_title("Thời gian huấn luyện (CPU)")
    _save(fig, path)


def fig_curves(logs_dir, path, items):
    fig, axs = plt.subplots(1, len(items), figsize=(5.5 * len(items), 4), squeeze=False)
    for ax, (exp, tag, title) in zip(axs[0], items):
        d = pd.read_csv(Path(logs_dir) / f"{exp}_{tag}_s0.csv")
        ax.plot(d.epoch, d.train_loss, color=COL["train"], label="train")
        ax.plot(d.epoch, d.val_loss, color=COL["val"], label="validation")
        b = d.val_loss.idxmin()
        ax.scatter([d.epoch[b]], [d.val_loss[b]], c="k", zorder=5, label=f"checkpoint (epoch {int(d.epoch[b])})")
        ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE"); ax.set_title(title); ax.legend()
    _save(fig, path)


def fig_final(state, activation, splits, metrics, fig_dir, title_tag):
    te = splits["test"]
    X, y = D.to_arrays(te)
    yh = numpy_forward(state, X, activation)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    axs[0].scatter(y, yh, s=18, c=COL["test"], alpha=0.8)
    lim = [0, max(y.max(), yh.max()) * 1.05]
    axs[0].plot(lim, lim, "k--", lw=1, label="$\\hat y = y$")
    axs[0].set_xlabel("y thật"); axs[0].set_ylabel("$\\hat y$ dự đoán"); axs[0].legend()
    axs[0].set_title(f"Tập test (100 mẫu): MSE={metrics['mse']:.2e}, MAE={metrics['mae']:.2e}")
    err = yh - y
    axs[1].scatter(y, err, s=18, c=COL["val"], alpha=0.8); axs[1].axhline(0, color="k", lw=1)
    axs[1].set_xlabel("y thật"); axs[1].set_ylabel("sai số $\\hat y - y$"); axs[1].set_title("Sai số theo giá trị y")
    _save(fig, Path(fig_dir) / "fig11_final_y_vs_yhat.png")

    G1, G2, P = _grid()
    Z = D.gaussian_pdf(P).reshape(G1.shape)
    Zh = numpy_forward(state, P / D.SCALE, activation).reshape(G1.shape)
    fig = plt.figure(figsize=(15, 4.6))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.plot_surface(G1, G2, Z, cmap="viridis", linewidth=0, alpha=0.85)
    ax.set_title("Hàm thật $f(x)$"); ax.view_init(28, -55)
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    ax.plot_surface(G1, G2, Zh, cmap="plasma", linewidth=0, alpha=0.85)
    ax.scatter(te.x1, te.x2, te.y, c="k", s=8, label="test (y thật)")
    ax.set_title("Bề mặt MLP xấp xỉ $\\hat f(x)$"); ax.view_init(28, -55); ax.legend(loc="upper left")
    for a in fig.axes:
        a.set_xlabel("$x_1$"); a.set_ylabel("$x_2$")
    ax = fig.add_subplot(1, 3, 3)
    E = np.abs(Zh - Z)
    cs = ax.contourf(G1, G2, E, 20, cmap="magma")
    fig.colorbar(cs, ax=ax, label="$|\\hat f - f|$")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_title(f"Sai số tuyệt đối trên lưới ({title_tag})")
    _save(fig, Path(fig_dir) / "fig12_final_surface.png")
    return dict(max_abs_err=float(E.max()), mean_abs_err=float(E.mean()))


def fig_arch(fig_dir):
    """Sơ đồ kiến trúc: (a) 2-B-1 một tầng ẩn, (b) C tầng ẩn."""
    def draw(ax, sizes, labels, title):
        xs = np.linspace(0, 1, len(sizes))
        pos = []
        for x, n in zip(xs, sizes):
            ys = np.linspace(0.12, 0.88, n) if n > 1 else [0.5]
            pos.append([(x, y) for y in ys])
        for L in range(len(sizes) - 1):
            for p in pos[L]:
                for q in pos[L + 1]:
                    ax.plot([p[0], q[0]], [p[1], q[1]], color="#999", lw=0.5, zorder=1)
        for L, pts in enumerate(pos):
            c = "#4c78a8" if L == 0 else ("#d1495b" if L == len(sizes) - 1 else "#f2a541")
            for p in pts:
                ax.scatter(*p, s=260, c=c, edgecolor="k", zorder=2)
            ax.text(xs[L], 0.0, labels[L], ha="center", va="top", fontsize=9)
        ax.set_xlim(-0.12, 1.12); ax.set_ylim(-0.2, 1.0); ax.axis("off"); ax.set_title(title)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    draw(axs[0], [2, 5, 1], ["đầu vào\n$(x_1,x_2)$", "tầng ẩn\n(B nơ-ron, tanh)", "đầu ra $\\hat y$\n(tuyến tính)"],
         "(a) MLP 1 tầng ẩn: 2 - B - 1 (minh hoạ B=5)")
    draw(axs[1], [2, 4, 4, 4, 1], ["đầu vào", "ẩn 1", "ẩn 2", "ẩn C", "đầu ra"],
         "(b) MLP C tầng ẩn (minh hoạ C=3, 4 nơ-ron/tầng)")
    for a in axs:
        a.grid(False)
    _save(fig, Path(fig_dir) / "fig10_architecture.png")
