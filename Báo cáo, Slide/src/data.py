"""Sinh / lưu / nạp dữ liệu Gaussian 2 chiều.

f(x) = 1/(2*pi*sqrt(det C)) * exp(-0.5 (x-mu)^T C^-1 (x-mu)),  mu=[0,0], C=[[0.3,-0.5],[-0.5,2]]
x1 ~ U(-2,2), x2 ~ U(-4,4).  Train/val/test dùng 3 luồng RNG độc lập (SeedSequence.spawn).
Tập train "pool" 2000 mẫu; thí nghiệm A lấy A mẫu đầu (tập con lồng nhau) -> so sánh công bằng.
"""
from pathlib import Path
import numpy as np
import pandas as pd

MU = np.array([0.0, 0.0])
COV = np.array([[0.3, -0.5], [-0.5, 2.0]])
X1_RANGE = (-2.0, 2.0)
X2_RANGE = (-4.0, 4.0)
DATA_SEED = 2026
N_POOL, N_VAL, N_TEST = 2000, 200, 100
SCALE = np.array([2.0, 4.0])  # chuẩn hoá theo miền: x / SCALE ∈ [-1,1] (không dùng thống kê test -> không rò rỉ)


def gaussian_pdf(x, mu=MU, cov=COV):
    """Gaussian nhiều chiều viết trực tiếp theo công thức đề bài."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    d = x.shape[1]
    diff = x - mu
    maha = np.einsum("ni,ij,nj->n", diff, np.linalg.inv(cov), diff)
    return np.exp(-0.5 * maha) / (((2 * np.pi) ** (d / 2)) * np.sqrt(np.linalg.det(cov)))


def sample_x(n, rng):
    x1 = rng.uniform(*X1_RANGE, size=n)
    x2 = rng.uniform(*X2_RANGE, size=n)
    return np.stack([x1, x2], axis=1)


def generate_splits(seed=DATA_SEED, n_pool=N_POOL, n_val=N_VAL, n_test=N_TEST):
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    out = {}
    for name, n, rng in zip(["train", "val", "test"], [n_pool, n_val, n_test], streams):
        x = sample_x(n, rng)
        out[name] = pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "y": gaussian_pdf(x)})
    return out


def save_splits(splits, data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for k, df in splits.items():
        df.to_csv(data_dir / f"{k}.csv", index=False)


def load_splits(data_dir):
    """Nạp CSV nếu có; nếu chưa có thì sinh (cùng seed) và lưu."""
    data_dir = Path(data_dir)
    if not all((data_dir / f"{k}.csv").exists() for k in ["train", "val", "test"]):
        save_splits(generate_splits(), data_dir)
    return {k: pd.read_csv(data_dir / f"{k}.csv") for k in ["train", "val", "test"]}


def to_arrays(df, n=None):
    """DataFrame -> (X chuẩn hoá, y). n: lấy n mẫu đầu."""
    if n is not None:
        df = df.iloc[:n]
    X = df[["x1", "x2"]].to_numpy(dtype=np.float64) / SCALE
    return X, df["y"].to_numpy(dtype=np.float64)


def describe(splits):
    rows = []
    for k, df in splits.items():
        for c in ["x1", "x2", "y"]:
            s = df[c]
            rows.append(dict(split=k, var=c, n=len(s), mean=s.mean(), std=s.std(), min=s.min(), max=s.max()))
    return pd.DataFrame(rows)
