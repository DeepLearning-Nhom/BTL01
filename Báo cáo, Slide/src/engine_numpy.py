"""Engine tham chiếu bằng NumPy (backprop viết tay + Adam), cùng thuật toán/khởi tạo với engine PyTorch.

Mục đích: kiểm chứng thuật toán (gradient check) và chạy được khi môi trường không có PyTorch.
Khởi tạo như nn.Linear mặc định: W,b ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in)). Loss = MSE trung bình theo batch.
"""
import time
import numpy as np
from .common import ACTS, regression_metrics

NAME = "numpy-reference"
_DACT = {  # đạo hàm theo đầu ra a = act(z)
    "tanh": lambda a: 1.0 - a * a,
    "relu": lambda a: (a > 0).astype(np.float64),
    "sigmoid": lambda a: a * (1.0 - a),
}


def init_params(sizes, rng):
    W, b = [], []
    for fi, fo in zip(sizes[:-1], sizes[1:]):
        bound = 1.0 / np.sqrt(fi)
        W.append(rng.uniform(-bound, bound, (fo, fi)))
        b.append(rng.uniform(-bound, bound, fo))
    return W, b


def forward(W, b, X, act, cache=False):
    a, acts = X, [X]
    for i in range(len(W)):
        z = a @ W[i].T + b[i]
        a = act(z) if i < len(W) - 1 else z
        acts.append(a)
    return (a, acts) if cache else a


def backward(W, acts, y, dact):
    """MSE mean: L = mean((pred-y)^2). Trả về dW, db."""
    n = y.shape[0]
    d = 2.0 * (acts[-1] - y) / n
    dW, db = [None] * len(W), [None] * len(W)
    for i in reversed(range(len(W))):
        dW[i] = d.T @ acts[i]
        db[i] = d.sum(axis=0)
        if i > 0:
            d = (d @ W[i]) * dact(acts[i])
    return dW, db


def gradient_check(seed=0, eps=1e-6):
    """Sai số tương đối giữa gradient giải tích và sai phân hữu hạn (kỳ vọng ~1e-8)."""
    rng = np.random.default_rng(seed)
    W, b = init_params([2, 5, 4, 1], rng)
    X, y = rng.normal(size=(8, 2)), rng.normal(size=(8, 1))
    act, dact = ACTS["tanh"], _DACT["tanh"]
    loss = lambda: float(np.mean((forward(W, b, X, act) - y) ** 2))
    _, acts = forward(W, b, X, act, cache=True)
    dW, db = backward(W, acts, y, dact)
    worst = 0.0
    for P, G in [(W, dW), (b, db)]:
        for p, g in zip(P, G):
            for idx in np.ndindex(p.shape):
                old = p[idx]
                p[idx] = old + eps; lp = loss()
                p[idx] = old - eps; lm = loss()
                p[idx] = old
                num = (lp - lm) / (2 * eps)
                worst = max(worst, abs(num - g[idx]) / max(1e-12, abs(num) + abs(g[idx])))
    return worst


def train_run(Xtr, ytr, Xva, yva, Xte, yte, hidden, activation="tanh", lr=1e-2,
              batch_size=64, epochs=300, patience=60, seed=0, device="cpu"):
    rng = np.random.default_rng(seed)
    act, dact = ACTS[activation], _DACT[activation]
    W, b = init_params([Xtr.shape[1], *hidden, 1], rng)
    params = W + b
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    b1, b2, eps, step = 0.9, 0.999, 1e-8, 0
    ytr_c, yva_c = ytr.reshape(-1, 1), yva.reshape(-1, 1)
    N = len(Xtr)

    history, best_val, best_epoch, best, bad = [], np.inf, 0, None, 0
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        perm = rng.permutation(N)
        for s in range(0, N, batch_size):
            idx = perm[s:s + batch_size]
            _, acts = forward(W, b, Xtr[idx], act, cache=True)
            dW, db = backward(W, acts, ytr_c[idx], dact)
            step += 1
            c1, c2 = 1 - b1 ** step, 1 - b2 ** step
            for k, (p, g) in enumerate(zip(params, dW + db)):
                m[k] = b1 * m[k] + (1 - b1) * g
                v[k] = b2 * v[k] + (1 - b2) * g * g
                p -= lr * (m[k] / c1) / (np.sqrt(v[k] / c2) + eps)   # in-place: cập nhật W, b
        train_loss = float(np.mean((forward(W, b, Xtr, act) - ytr_c) ** 2))
        val_loss = float(np.mean((forward(W, b, Xva, act) - yva_c) ** 2))
        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss))
        if val_loss < best_val - 1e-12:
            best_val, best_epoch, bad = val_loss, epoch, 0
            best = ([w.copy() for w in W], [x.copy() for x in b])
        else:
            bad += 1
            if bad >= patience:
                break
    train_time = time.perf_counter() - t0

    W, b = best
    state = {}
    for i in range(len(W)):
        state[f"W{i}"], state[f"b{i}"] = W[i], b[i]
    metrics = {k: regression_metrics(y, forward(W, b, X, act).reshape(-1))
               for k, (X, y) in {"train": (Xtr, ytr), "val": (Xva, yva), "test": (Xte, yte)}.items()}
    return dict(history=history, best_epoch=best_epoch, epochs_run=len(history), train_time_s=train_time,
                state=state, metrics=metrics)


def save_checkpoint(res, path, meta):
    np.savez(str(path) + ".npz", **res["state"])
