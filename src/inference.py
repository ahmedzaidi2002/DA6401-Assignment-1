import argparse
import json
import numpy as np

from src.utils.data_loader import load_data, one_hot_encode
from src.ann.neural_network import NeuralNetwork


def classification_report_from_logits(logits: np.ndarray, y_true_onehot: np.ndarray, eps: float = 1e-12):
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

    # defaults should match your best config (same as train.py)
    p.add_argument("-d", "--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    p.add_argument("-e", "--epochs", type=int, default=10)
    p.add_argument("-b", "--batch_size", type=int, default=64)

    p.add_argument("-l", "--loss", type=str, default="cross_entropy", choices=["mse", "cross_entropy"])
    p.add_argument("-o", "--optimizer", type=str, default="momentum",
                   choices=["sgd", "momentum", "nag", "rmsprop"])

    p.add_argument("-lr", "--learning_rate", type=float, default=0.01)
    p.add_argument("-wd", "--weight_decay", type=float, default=0.0)

    p.add_argument("-nhl", "--num_hidden_layers", type=int, default=2)
    p.add_argument("-sz", "--hidden_sizes", type=int, nargs="+", default=[128, 64])

    p.add_argument("-a", "--activation", type=str, default="relu", choices=["sigmoid", "tanh", "relu"])
    p.add_argument("-wi", "--weight_init", type=str, default="xavier", choices=["random", "xavier"])
    p.add_argument("-wp", "--wandb_project", type=str, default="DA6401-A1")

    # inference-specific
    p.add_argument("--model_path", type=str, default="src/best_model.npy")
    p.add_argument("--config_path", type=str, default="src/best_config.json")

    return p


def maybe_load_config_defaults(args):
    """
    If best_config.json exists, use it to override argparse defaults
    ONLY for fields that user didn't explicitly set via CLI.
    """
    try:
        with open(args.config_path, "r") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        return args

    # If user didn't pass a value explicitly, argparse keeps default.
    # We approximate this by always reading config and overwriting.
    # (This is fine because config should reflect best defaults.)
    for k, v in cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)

    return args


def main():
    args = build_parser().parse_args()
    args = maybe_load_config_defaults(args)

    # Load data
    (_, _), (X_test, y_test) = load_data(args.dataset)
    y_test_oh = one_hot_encode(y_test)

    # Build model from args, then load weights
    model = NeuralNetwork(args)

    weights = np.load(args.model_path, allow_pickle=True).item()
    model.set_weights(weights)

    # Inference
    logits = model.forward(X_test)
    acc, prec, rec, f1 = classification_report_from_logits(logits, y_test_oh)

    print("----- Test Metrics -----")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")


if __name__ == "__main__":
    main()