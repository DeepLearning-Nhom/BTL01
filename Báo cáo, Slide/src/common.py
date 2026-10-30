"""Hằng số + hàm dùng chung (không phụ thuộc PyTorch)."""
import numpy as np

ACTS = {
    "tanh": np.tanh,
    "relu": lambda z: np.maximum(z, 0.0),
    "sigmoid": lambda z: 1.0 / (1.0 + np.exp(-z)),
}


def numpy_forward(state, X, activation="tanh"):
    """Forward bằng NumPy từ state {W0,b0,W1,b1,...}; W có shape (out,in) như nn.Linear.
    Dùng để vẽ bề mặt xấp xỉ độc lập với engine huấn luyện."""
    n_layers = len([k for k in state if k.startswith("W")])
    a = np.asarray(X, dtype=np.float64)
    act = ACTS[activation]
    for i in range(n_layers):
        a = a @ np.asarray(state[f"W{i}"], dtype=np.float64).T + np.asarray(state[f"b{i}"], dtype=np.float64)
        if i < n_layers - 1:
            a = act(a)
    return a.reshape(-1)


def regression_metrics(y_true, y_pred):
    e = np.asarray(y_pred, dtype=np.float64).reshape(-1) - np.asarray(y_true, dtype=np.float64).reshape(-1)
    mse = float(np.mean(e ** 2))
    return {"mse": mse, "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(mse))}
