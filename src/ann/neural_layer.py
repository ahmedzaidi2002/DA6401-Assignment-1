import numpy as np


class Layer:
    """Fully-connected layer: Z = XW + b"""

    def __init__(self, input_size: int, output_size: int, weight_init: str = "random"):
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        scheme = (weight_init or "").strip().lower()

        self.W, self.b = self._init_params(self.input_size, self.output_size, scheme)

        self._x = None  # cache input for backward

        # autograder-required gradient fields
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

    @staticmethod
    def _init_params(in_dim: int, out_dim: int, scheme: str):
        if scheme == "xavier":
            limit = np.sqrt(6.0 / (in_dim + out_dim))
            W = np.random.uniform(-limit, limit, size=(in_dim, out_dim))
        elif scheme == "random":
            W = 0.01 * np.random.randn(in_dim, out_dim)
        else:
            raise ValueError(f"weight_init must be 'random' or 'xavier', got '{scheme}'")

        b = np.zeros((1, out_dim), dtype=float)
        return W.astype(float), b

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2 or x.shape[1] != self.W.shape[0]:
            raise ValueError(f"Expected x shape (batch, {self.W.shape[0]}), got {x.shape}")

        self._x = x
        return x @ self.W + self.b

    def backward(self, dZ: np.ndarray) -> np.ndarray:
        if self._x is None:
            raise RuntimeError("Layer.backward called before forward().")
        if dZ.ndim != 2 or dZ.shape[1] != self.W.shape[1]:
            raise ValueError(f"Expected dZ shape (batch, {self.W.shape[1]}), got {dZ.shape}")

        m = self._x.shape[0]
        self.grad_W = (self._x.T @ dZ) / m
        self.grad_b = np.sum(dZ, axis=0, keepdims=True) / m
        return dZ @ self.W.T