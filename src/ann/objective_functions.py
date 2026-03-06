import numpy as np


class Loss:
    """
    Loss functions.

    Supported:
      - mse            (expects logits/raw outputs)
      - cross_entropy  (expects logits; applies softmax internally)

    Gradient scaling rule used here:
      - Loss.compute() returns mean over batch.
      - Loss.backward() returns d_logits WITHOUT dividing by batch size (m),
        because Layer.backward() already divides by m.

    For MSE, we include the correct factor for mean over classes (C).
    """

    def __init__(self, name: str):
        self.name = (name or "").strip().lower()
        if self.name not in {"mse", "cross_entropy"}:
            raise ValueError("Loss must be 'mse' or 'cross_entropy'")

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        z = logits - np.max(logits, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def compute(self, y_true: np.ndarray, logits: np.ndarray) -> float:
        m = y_true.shape[0]

        if self.name == "mse":
            # mean over all elements
            return float(np.mean((y_true - logits) ** 2))

        probs = self._softmax(logits)
        eps = 1e-12
        return float(-np.sum(y_true * np.log(probs + eps)) / m)

    def backward(self, y_true: np.ndarray, logits: np.ndarray) -> np.ndarray:
        """
        Returns d_logits (same shape as logits).
        NO division by batch size here.
        """
        if self.name == "mse":
            # If compute() is mean over all elements:
            # L = mean((y - logits)^2) -> dL/dlogits = 2*(logits - y)/(m*C)
            # Layer.backward divides by m, so we return: 2*(logits-y)/C
            C = y_true.shape[1]
            return (2.0 / C) * (logits - y_true)

        probs = self._softmax(logits)
        return (probs - y_true)