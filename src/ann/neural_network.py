"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np

from .neural_layer import Layer
from .activations import Activation
from .objective_functions import Loss


class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, cli_args):
        self.args = cli_args

        # ----- infer dimensions -----
        dataset = getattr(cli_args, "dataset", "mnist")
        self.input_dim = 784  # MNIST/Fashion-MNIST = 28*28
        self.output_dim = 10

        # ----- architecture from CLI -----
        n_hidden = int(getattr(cli_args, "num_hidden_layers", 1))
        hidden_sizes = list(getattr(cli_args, "hidden_sizes", [128]))

        # Make it robust even if someone passes fewer sizes than layers
        if len(hidden_sizes) < n_hidden:
            hidden_sizes = hidden_sizes + [hidden_sizes[-1]] * (n_hidden - len(hidden_sizes))
        hidden_sizes = hidden_sizes[:n_hidden]

        activation_name = getattr(cli_args, "activation", "relu")
        weight_init = getattr(cli_args, "weight_init", "random")

        layer_sizes = [self.input_dim] + hidden_sizes + [self.output_dim]

        # ----- build layers and activations -----
        self.layers = []
        self.activations = []

        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.layers.append(Layer(in_dim, out_dim, weight_init=weight_init))

        # activations only for hidden layers (not for final logits)
        for _ in range(len(self.layers) - 1):
            self.activations.append(Activation(activation_name))

        # ----- loss -----
        self.loss_fn = Loss(getattr(cli_args, "loss", "cross_entropy"))

        # ----- optimizer hyperparams (optimizer object can be plugged later) -----
        self.lr = float(getattr(cli_args, "learning_rate", 1e-3))
        self.weight_decay = float(getattr(cli_args, "weight_decay", 0.0))
        self.optimizer = None  # set from train.py (sgd/momentum/nag/rmsprop)

        # grads stored after backward (object arrays, index 0 = last layer)
        self.grad_W = None
        self.grad_b = None

    def forward(self, X):
        """
        Forward propagation through all layers.
        Returns logits (no softmax applied)
        X is shape (b, D_in) and output is shape (b, D_out).
        b is batch size, D_in is input dimension, D_out is output dimension.
        """
        a = X
        for i, layer in enumerate(self.layers):
            z = layer.forward(a)
            if i < len(self.activations):
                a = self.activations[i].forward(z)
            else:
                a = z  # final layer -> logits
        return a

    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        Returns two numpy arrays: grad_Ws, grad_bs.
        - `grad_Ws[0]` is gradient for the last (output) layer weights,
          `grad_bs[0]` is gradient for the last layer biases, and so on.
        """
        # y_pred here is logits (by updated instruction)
        d_logits = self.loss_fn.backward(y_true, y_pred)

        grad_W_list = []
        grad_b_list = []

        grad = d_logits

        # Backprop through layers in reverse; collect grads so that index 0 = last layer
        for layer_idx in range(len(self.layers) - 1, -1, -1):
            # linear backward sets layer.grad_W and layer.grad_b
            grad = self.layers[layer_idx].backward(grad)
            grad_W_list.append(self.layers[layer_idx].grad_W)
            grad_b_list.append(self.layers[layer_idx].grad_b)

            # activation backward for previous hidden layer (if any)
            act_idx = layer_idx - 1
            if act_idx >= 0:
                grad = self.activations[act_idx].backward(grad)

        # create explicit object arrays to avoid numpy trying to broadcast shapes
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b

    def update_weights(self):
        """
        Applies parameter updates using the optimizer.
        train.py should set self.optimizer to an object with a step(layers) method
        OR you can keep this as vanilla SGD fallback.
        """
        if self.optimizer is not None:
            # recommended: optimizer handles momentum/rmsprop/nag buffers internally
            self.optimizer.step(self.layers)
            return

        # Fallback: plain SGD with optional weight decay (L2)
        for layer in self.layers:
            if self.weight_decay != 0.0:
                layer.W *= (1.0 - self.lr * self.weight_decay)
            layer.W -= self.lr * layer.grad_W
            layer.b -= self.lr * layer.grad_b

    def train(self, X_train, y_train, epochs=1, batch_size=32):
        """
        Minimal training loop (your train.py can be richer + wandb logging).
        Assumes y_train is one-hot.
        """
        n = X_train.shape[0]
        for _ in range(int(epochs)):
            perm = np.random.permutation(n)
            X_train = X_train[perm]
            y_train = y_train[perm]

            for start in range(0, n, int(batch_size)):
                end = start + int(batch_size)
                Xb = X_train[start:end]
                yb = y_train[start:end]

                logits = self.forward(Xb)
                _ = self.loss_fn.compute(yb, logits)  # compute if you want it logged
                self.backward(yb, logits)
                self.update_weights()

    def evaluate(self, X, y):
        """
        Returns accuracy on given set.
        Assumes y is one-hot.
        """
        logits = self.forward(X)
        y_pred = np.argmax(logits, axis=1)
        y_true = np.argmax(y, axis=1)
        return float(np.mean(y_pred == y_true))

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()