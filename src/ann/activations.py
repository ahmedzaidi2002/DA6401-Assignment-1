import numpy as np


class Activation:
    """
    Activation function module.

    Supported:
      - sigmoid
      - tanh
      - relu
      - softmax (usually used inside loss, not as final layer activation)

    Note:
      The assignment requires the network to output logits, so do NOT apply softmax
      in the model forward for the final layer.
    """

    def __init__(self, name: str):
        self.name = (name or "").strip().lower()
        self._z = None
        self._a = None

        supported = {"sigmoid", "tanh", "relu", "softmax"}
        if self.name not in supported:
            raise ValueError(f"Unsupported activation '{name}'. Supported: {sorted(supported)}")

    def forward(self, z: np.ndarray) -> np.ndarray:
        self._z = z

        if self.name == "sigmoid":
            out = np.empty_like(z, dtype=float)
            pos = z >= 0
            neg = ~pos
            out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
            ez = np.exp(z[neg])
            out[neg] = ez / (1.0 + ez)
            self._a = out

        elif self.name == "tanh":
            self._a = np.tanh(z)

        elif self.name == "relu":
            self._a = np.maximum(0.0, z)

        elif self.name == "softmax":
            z_shift = z - np.max(z, axis=1, keepdims=True)
            exp_z = np.exp(z_shift)
            self._a = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        return self._a

    def backward(self, dA: np.ndarray) -> np.ndarray:
        if self._a is None or self._z is None:
            raise RuntimeError("Activation.backward() called before forward().")

        if self.name == "sigmoid":
            return dA * (self._a * (1.0 - self._a))

        if self.name == "tanh":
            return dA * (1.0 - self._a ** 2)

        if self.name == "relu":
            return dA * (self._z > 0)

        # softmax general vector-Jacobian product
        s = self._a
        dot = np.sum(dA * s, axis=1, keepdims=True)
        return s * (dA - dot)