import numpy as np


class Optimizer:
    """
    Assignment optimizers:
      - sgd
      - momentum
      - nag
      - rmsprop
    """

    def __init__(self, name="sgd", learning_rate=1e-3, weight_decay=0.0):
        self.name = (name or "").strip().lower()
        self.lr = float(learning_rate)
        self.wd = float(weight_decay)

        if self.name not in {"sgd", "momentum", "nag", "rmsprop"}:
            raise ValueError("Optimizer must be one of: sgd, momentum, nag, rmsprop")

        self.beta = 0.9
        self.eps = 1e-8

    def step(self, layers):  # <- renamed from update to step
        for layer in layers:
            gW = layer.grad_W
            gb = layer.grad_b

            if self.wd != 0.0:
                gW = gW + self.wd * layer.W

            if self.name == "sgd":
                layer.W -= self.lr * gW
                layer.b -= self.lr * gb
                continue

            if self.name in {"momentum", "nag"}:
                if not hasattr(layer, "vW"):
                    layer.vW = np.zeros_like(layer.W)
                    layer.vb = np.zeros_like(layer.b)

                vW_prev = layer.vW.copy()
                vb_prev = layer.vb.copy()

                layer.vW = self.beta * layer.vW + gW
                layer.vb = self.beta * layer.vb + gb

                if self.name == "momentum":
                    layer.W -= self.lr * layer.vW
                    layer.b -= self.lr * layer.vb
                else:
                    layer.W -= self.lr * (self.beta * vW_prev + gW)
                    layer.b -= self.lr * (self.beta * vb_prev + gb)
                continue

            if not hasattr(layer, "sW"):
                layer.sW = np.zeros_like(layer.W)
                layer.sb = np.zeros_like(layer.b)

            layer.sW = self.beta * layer.sW + (1.0 - self.beta) * (gW ** 2)
            layer.sb = self.beta * layer.sb + (1.0 - self.beta) * (gb ** 2)

            layer.W -= self.lr * gW / (np.sqrt(layer.sW) + self.eps)
            layer.b -= self.lr * gb / (np.sqrt(layer.sb) + self.eps)