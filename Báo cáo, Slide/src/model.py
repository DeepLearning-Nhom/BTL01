"""MLP tuỳ biến số tầng ẩn / số nơ-ron bằng torch.nn.Module."""
import torch
from torch import nn

ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "sigmoid": nn.Sigmoid}


class MLP(nn.Module):
    """in_dim -> [Linear + act] x len(hidden) -> Linear(out_dim). Đầu ra tuyến tính (bài toán hồi quy)."""

    def __init__(self, in_dim=2, hidden=(10,), out_dim=1, activation="tanh"):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), ACTIVATIONS[activation]()]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
