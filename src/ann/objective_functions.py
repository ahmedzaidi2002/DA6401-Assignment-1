import numpy as np


class Loss:
    """
    Loss functions for the neural network.

    Supported:
      - mse            (Mean Squared Error — expects raw logits)
      - cross_entropy  (Cross-Entropy — expects logits; applies softmax internally)

    Convention:
      - compute() returns the mean loss over the batch.
      - backward() returns dL/dlogits already divided by batch size m,
        so that downstream layers can compute grad_W = X.T @ dZ directly
        without needing to know the batch size.
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
        """Return scalar mean loss over the batch."""
        m = y_true.shape[0]

        if self.name == "mse":
            return float(np.mean((y_true - logits) ** 2))

        probs = self._softmax(logits)
        eps = 1e-12
        return float(-np.sum(y_true * np.log(probs + eps)) / m)

    def backward(self, y_true: np.ndarray, logits: np.ndarray) -> np.ndarray:
        """
        Returns dL/dlogits (same shape as logits), divided by batch size m.

        This is the standard convention: the loss gradient is fully normalized,
        so layers can simply compute grad_W = X.T @ dZ without extra scaling.
        """
        m = y_true.shape[0]

        if self.name == "mse":
            # L = 1/(m*C) * sum((y - logits)^2)
            # dL/dlogits = 2*(logits - y) / (m*C)
            C = y_true.shape[1]
            return 2.0 * (logits - y_true) / (m * C)

        # Cross-entropy with softmax
        probs = self._softmax(logits)
        return (probs - y_true) / m