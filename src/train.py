import argparse
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.data_loader import load_data, one_hot_encode
from src.ann.neural_network import NeuralNetwork
from src.ann.optimizers import Optimizer


def classification_report_from_logits(logits: np.ndarray, y_true_onehot: np.ndarray, eps: float = 1e-12):
    """
    Returns (accuracy, precision, recall, f1) using macro-averaging.
    y_true_onehot: (N, C)
    logits: (N, C)
    """
    y_true = np.argmax(y_true_onehot, axis=1)
    y_pred = np.argmax(logits, axis=1)
    C = y_true_onehot.shape[1]

    acc = float(np.mean(y_pred == y_true))

    f1s, precs, recs = [], [], []
    for k in range(C):
        tp = np.sum((y_pred == k) & (y_true == k))
        fp = np.sum((y_pred == k) & (y_true != k))
        fn = np.sum((y_pred != k) & (y_true == k))

        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)

        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)

    return acc, float(np.mean(precs)), float(np.mean(recs)), float(np.mean(f1s))


def build_parser():
    p = argparse.ArgumentParser()

    # Use best config as defaults (fill these after your sweeps)
    p.add_argument("-d", "--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    p.add_argument("-e", "--epochs", type=int, default=10)
    p.add_argument("-b", "--batch_size", type=int, default=64)

    p.add_argument("-l", "--loss", type=str, default="cross_entropy", choices=["mse", "cross_entropy"])
    p.add_argument("-o", "--optimizer", type=str, default="momentum",
                   choices=["sgd", "momentum", "nag", "rmsprop"])

    p.add_argument("-lr", "--learning_rate", type=float, default=0.01)
    p.add_argument("-wd", "--weight_decay", type=float, default=0.0)

    # updated naming: num_hidden_layers + hidden_sizes
    p.add_argument("-nhl", "--num_hidden_layers", type=int, default=2)
    p.add_argument("-sz", "--hidden_sizes", type=int, nargs="+", default=[128, 64])

    p.add_argument("-a", "--activation", type=str, default="relu", choices=["sigmoid", "tanh", "relu"])
    p.add_argument("-wi", "--weight_init", type=str, default="xavier", choices=["random", "xavier"])

    # update requires this
    p.add_argument("-wp", "--wandb_project", type=str, default="Assignment_1")

    # save paths (must end up in src/)
    p.add_argument("--save_model", type=str, default=os.path.join("src", "best_model.npy"))
    p.add_argument("--save_config", type=str, default=os.path.join("src", "best_config.json"))

    return p


def main():
    args = build_parser().parse_args()

    # ---------------- data ----------------
    (X_all, y_all), (X_test, y_test) = load_data(args.dataset)

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.1, random_state=42, stratify=y_all
    )

    y_train_oh = one_hot_encode(y_train)
    y_val_oh = one_hot_encode(y_val)
    y_test_oh = one_hot_encode(y_test)

    # ---------------- model ----------------
    model = NeuralNetwork(args)
    model.optimizer = Optimizer(args.optimizer, args.learning_rate, args.weight_decay)

    # ---------------- training ----------------
    best_test_f1 = -1.0
    n = X_train.shape[0]

    for epoch in range(int(args.epochs)):
        perm = np.random.permutation(n)
        X_train = X_train[perm]
        y_train_oh = y_train_oh[perm]

        for start in range(0, n, int(args.batch_size)):
            xb = X_train[start:start + int(args.batch_size)]
            yb = y_train_oh[start:start + int(args.batch_size)]

            logits = model.forward(xb)
            model.backward(yb, logits)
            model.update_weights()

        # ---- evaluate each epoch ----
        val_logits = model.forward(X_val)
        test_logits = model.forward(X_test)

        val_acc, val_p, val_r, val_f1 = classification_report_from_logits(val_logits, y_val_oh)
        test_acc, test_p, test_r, test_f1 = classification_report_from_logits(test_logits, y_test_oh)

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Val: acc={val_acc:.4f}, f1={val_f1:.4f} | "
            f"Test: acc={test_acc:.4f}, f1={test_f1:.4f}"
        )

        # Save best by TEST F1 (per instructions)
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1

            os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
            np.save(args.save_model, model.get_weights())

            os.makedirs(os.path.dirname(args.save_config), exist_ok=True)
            with open(args.save_config, "w") as f:
                json.dump(vars(args), f, indent=2)

    print(f"Done. Best Test F1 = {best_test_f1:.4f}")
    print(f"Saved model to: {args.save_model}")
    print(f"Saved config to: {args.save_config}")


if __name__ == "__main__":
    main()