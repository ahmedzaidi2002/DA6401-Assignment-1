"""
Main Neural Network Model class.
Handles forward and backward propagation loops.
"""
import numpy as np

try:
    from .neural_layer import Layer
    from .activations import Activation
    from .objective_functions import Loss
    from .optimizers import Optimizer
except ImportError:
    from neural_layer import Layer
    from activations import Activation
    from objective_functions import Loss
    from optimizers import Optimizer


class NeuralNetwork:
    def __init__(self, cli_args):
        self.args = cli_args

        self.activation_name = getattr(cli_args, "activation", "relu") or "relu"
        self.weight_init = getattr(cli_args, "weight_init", "random") or "random"
        loss_name = getattr(cli_args, "loss", "cross_entropy") or "cross_entropy"
        self.loss_fn = Loss(loss_name)

        self.lr = float(getattr(cli_args, "learning_rate", 1e-3) or 1e-3)
        self.weight_decay = float(getattr(cli_args, "weight_decay", 0.0) or 0.0)
        self.optimizer = None

        self.grad_W = None
        self.grad_b = None

        layer_sizes = self._resolve_layer_sizes(cli_args)
        self._build_network(layer_sizes)

    def _resolve_layer_sizes(self, cli_args):
        input_dim = getattr(cli_args, "input_dim", None)
        if input_dim is None:
            input_dim = getattr(cli_args, "input_size", 784)
        input_dim = int(input_dim) if input_dim is not None else 784

        output_dim = getattr(cli_args, "output_dim", None)
        if output_dim is None:
            output_dim = getattr(cli_args, "output_size", 10)
        output_dim = int(output_dim) if output_dim is not None else 10

        if hasattr(cli_args, "layer_sizes") and getattr(cli_args, "layer_sizes") is not None:
            raw_sizes = list(getattr(cli_args, "layer_sizes"))
            raw_sizes = [int(x) for x in raw_sizes]

            if len(raw_sizes) == 0:
                return [input_dim, output_dim]

            if raw_sizes[0] != input_dim:
                raw_sizes = [input_dim] + raw_sizes
            if raw_sizes[-1] != output_dim:
                raw_sizes = raw_sizes + [output_dim]

            return raw_sizes

        # hidden-layer interface
        n_hidden = getattr(cli_args, "num_hidden_layers", None)
        if n_hidden is None:
            n_hidden = getattr(cli_args, "num_layers", 1)
        n_hidden = int(n_hidden) if n_hidden is not None else 1

        hidden_sizes = getattr(cli_args, "hidden_sizes", None)
        if hidden_sizes is None:
            hidden_size = getattr(cli_args, "hidden_size", 128)

            if hidden_size is None:
                hidden_sizes = [128] * n_hidden
            elif np.isscalar(hidden_size):
                hidden_sizes = [int(hidden_size)] * n_hidden
            else:
                hidden_sizes = [int(x) for x in list(hidden_size)]
        else:
            hidden_sizes = [int(x) for x in list(hidden_sizes)]

        if n_hidden == 0:
            hidden_sizes = []
        else:
            if len(hidden_sizes) == 0:
                hidden_sizes = [128] * n_hidden
            elif len(hidden_sizes) < n_hidden:
                hidden_sizes = hidden_sizes + [hidden_sizes[-1]] * (n_hidden - len(hidden_sizes))
            else:
                hidden_sizes = hidden_sizes[:n_hidden]

        return [input_dim] + hidden_sizes + [output_dim]

    def _build_network(self, layer_sizes):
        self.layer_sizes = [int(x) for x in layer_sizes]
        self.layers = []
        self.activations = []

        for in_dim, out_dim in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            self.layers.append(Layer(in_dim, out_dim, weight_init=self.weight_init))

        for _ in range(len(self.layers) - 1):
            self.activations.append(Activation(self.activation_name))

    def forward(self, X):
        """Forward pass through all layers."""
        a = X
        for i, layer in enumerate(self.layers):
            z = layer.forward(a)
            if i < len(self.activations):
                a = self.activations[i].forward(z)
            else:
                a = z  # output layer: raw logits
        return a

    def backward(self, y_true, y_pred):
        """
        Backward pass: computes gradients for all layers.

        The loss function's backward returns dL/dlogits already normalized by 1/m.
        Each layer's backward stores grad_W = X.T @ dZ and grad_b = sum(dZ)
        without any additional normalization.
        """
        d_logits = self.loss_fn.backward(y_true, y_pred)
        grad = d_logits

        for layer_idx in range(len(self.layers) - 1, -1, -1):
            grad = self.layers[layer_idx].backward(grad)

            if layer_idx > 0:
                grad = self.activations[layer_idx - 1].backward(grad)

        # Collect gradients into arrays for external access
        self.grad_W = np.empty(len(self.layers), dtype=object)
        self.grad_b = np.empty(len(self.layers), dtype=object)

        for i, layer in enumerate(self.layers):
            self.grad_W[i] = layer.grad_W
            self.grad_b[i] = layer.grad_b

        return self.grad_W, self.grad_b

    def update_weights(self):
        """Update weights using the optimizer or simple SGD."""
        if self.optimizer is not None:
            self.optimizer.step(self.layers)
            return

        for layer in self.layers:
            if self.weight_decay != 0.0:
                layer.W *= (1.0 - self.lr * self.weight_decay)
            layer.W -= self.lr * layer.grad_W
            layer.b -= self.lr * layer.grad_b

    def train(self, X_train, y_train, epochs=1, batch_size=32):
        """Train the model for the specified number of epochs."""
        n = X_train.shape[0]

        for _ in range(int(epochs)):
            perm = np.random.permutation(n)
            X_shuf = X_train[perm]
            y_shuf = y_train[perm]

            for start in range(0, n, int(batch_size)):
                end = start + int(batch_size)
                xb = X_shuf[start:end]
                yb = y_shuf[start:end]
                logits = self.forward(xb)
                self.backward(yb, logits)
                self.update_weights()

    def evaluate(self, X, y):
        """Return accuracy on the given data."""
        logits = self.forward(X)
        y_pred = np.argmax(logits, axis=1)
        y_true = np.argmax(y, axis=1)
        return float(np.mean(y_pred == y_true))

    def get_weights(self):
        """Return a dict of all weight matrices and bias vectors."""
        weight_dict = {}
        for i, layer in enumerate(self.layers):
            weight_dict[f"W{i}"] = layer.W.copy()
            weight_dict[f"b{i}"] = layer.b.copy()
        return weight_dict

    def set_weights(self, weight_dict):
        """Load weights from a dict, rebuilding network if shapes differ."""
        weight_keys = sorted(
            [k for k in weight_dict.keys() if k.startswith("W")],
            key=lambda x: int(x[1:])
        )

        inferred_sizes = []
        for idx, key in enumerate(weight_keys):
            W = np.asarray(weight_dict[key], dtype=float)
            if idx == 0:
                inferred_sizes.append(W.shape[0])
            inferred_sizes.append(W.shape[1])

        rebuild_needed = False

        if len(inferred_sizes) != len(self.layer_sizes):
            rebuild_needed = True
        else:
            for i, layer in enumerate(self.layers):
                if layer.W.shape != np.asarray(weight_dict[f"W{i}"]).shape:
                    rebuild_needed = True
                    break

        if rebuild_needed:
            self._build_network(inferred_sizes)

        for i, layer in enumerate(self.layers):
            layer.W = np.asarray(weight_dict[f"W{i}"], dtype=float).copy()
            layer.b = np.asarray(weight_dict[f"b{i}"], dtype=float).reshape(1, -1).copy()
            layer.grad_W = np.zeros_like(layer.W)
            layer.grad_b = np.zeros_like(layer.b)