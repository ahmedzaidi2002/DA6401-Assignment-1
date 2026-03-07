import numpy as np


class Optimizer:
    """
    Optimizers for neural network training.

    Supported:
      - sgd       : Vanilla stochastic gradient descent
      - momentum  : SGD with momentum
      - nag       : Nesterov accelerated gradient
      - rmsprop   : RMSProp adaptive learning rate
    """

    def __init__(self, name="sgd", learning_rate=1e-3, weight_decay=0.0):
        self.name = (name or "").strip().lower()
        self.lr = float(learning_rate)
        self.wd = float(weight_decay)

        supported = {"sgd", "momentum", "nag", "rmsprop"}
        if self.name not in supported:
            raise ValueError(f"Optimizer must be one of: {sorted(supported)}")

        # Hyperparameters
        self.beta = 0.9       # momentum decay
        self.beta2 = 0.999    # second moment decay (rmsprop)
        self.eps = 1e-8

    def step(self, layers):
        """Perform one optimization step on all layers."""
        for layer in layers:
            gW = layer.grad_W
            gb = layer.grad_b

            # L2 regularization (weight decay)
            if self.wd != 0.0:
                gW = gW + self.wd * layer.W

            if self.name == "sgd":
                layer.W -= self.lr * gW
                layer.b -= self.lr * gb

            elif self.name == "momentum":
                if not hasattr(layer, "_vW"):
                    layer._vW = np.zeros_like(layer.W)
                    layer._vb = np.zeros_like(layer.b)

                layer._vW = self.beta * layer._vW + gW
                layer._vb = self.beta * layer._vb + gb

                layer.W -= self.lr * layer._vW
                layer.b -= self.lr * layer._vb

            elif self.name == "nag":
                if not hasattr(layer, "_vW"):
                    layer._vW = np.zeros_like(layer.W)
                    layer._vb = np.zeros_like(layer.b)

                vW_prev = layer._vW.copy()
                vb_prev = layer._vb.copy()

                layer._vW = self.beta * layer._vW + gW
                layer._vb = self.beta * layer._vb + gb

                layer.W -= self.lr * (self.beta * vW_prev + gW)
                layer.b -= self.lr * (self.beta * vb_prev + gb)

            elif self.name == "rmsprop":
                if not hasattr(layer, "_sW"):
                    layer._sW = np.zeros_like(layer.W)
                    layer._sb = np.zeros_like(layer.b)

                layer._sW = self.beta2 * layer._sW + (1.0 - self.beta2) * (gW ** 2)
                layer._sb = self.beta2 * layer._sb + (1.0 - self.beta2) * (gb ** 2)

                layer.W -= self.lr * gW / (np.sqrt(layer._sW) + self.eps)
                layer.b -= self.lr * gb / (np.sqrt(layer._sb) + self.eps)