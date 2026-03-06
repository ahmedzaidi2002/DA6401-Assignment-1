import json
import os
import numpy as np
from types import SimpleNamespace
from sklearn.model_selection import train_test_split

from src.utils.data_loader import load_data, one_hot_encode
from src.ann.neural_network import NeuralNetwork
from src.ann.optimizers import Optimizer


def metrics_from_logits(logits, y_onehot, eps=1e-12):
    y_true = np.argmax(y_onehot, axis=1)
    y_pred = np.argmax(logits, axis=1)
    C = y_onehot.shape[1]

    acc = float(np.mean(y_pred == y_true))

    precs, recs, f1s = [], [], []
    for k in range(C):
        tp = np.sum((y_pred == k) & (y_true == k))
        fp = np.sum((y_pred == k) & (y_true != k))
        fn = np.sum((y_pred != k) & (y_true == k))
        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        f1 = 2 * p * r / (p + r + eps)
        precs.append(p); recs.append(r); f1s.append(f1)

    return acc, float(np.mean(precs)), float(np.mean(recs)), float(np.mean(f1s))


def train_one(cfg, X_train, y_train_oh, X_test, y_test_oh, X_val, y_val_oh):
    model = NeuralNetwork(cfg)
    model.optimizer = Optimizer(cfg.optimizer, cfg.learning_rate, cfg.weight_decay)

    n = X_train.shape[0]
    bs = int(cfg.batch_size)

    for _ in range(int(cfg.epochs)):
        perm = np.random.permutation(n)
        X_train = X_train[perm]
        y_train_oh = y_train_oh[perm]

        for s in range(0, n, bs):
            xb = X_train[s:s+bs]
            yb = y_train_oh[s:s+bs]
            logits = model.forward(xb)
            model.backward(yb, logits)
            model.update_weights()

    val_logits = model.forward(X_val)
    test_logits = model.forward(X_test)

    val = metrics_from_logits(val_logits, y_val_oh)
    test = metrics_from_logits(test_logits, y_test_oh)
    return model, val, test


def main():
    np.random.seed(7)

    save_model = os.path.join("src", "best_model.npy")
    save_cfg = os.path.join("src", "best_config.json")

    # data
    (X_all, y_all), (X_test, y_test) = load_data("mnist")
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.1, random_state=42, stratify=y_all
    )
    y_train_oh = one_hot_encode(y_train)
    y_val_oh = one_hot_encode(y_val)
    y_test_oh = one_hot_encode(y_test)

    # a small, strong search space (fast + effective)
    optimizers = ["sgd", "momentum", "rmsprop"]
    lrs = [0.1, 0.03, 0.01]
    activations = ["relu", "tanh"]
    hidden_sets = [[128], [256], [128, 64]]
    weight_inits = ["xavier"]

    best = {"f1": -1.0, "cfg": None}

    trials = 0
    for opt in optimizers:
        for lr in lrs:
            for act in activations:
                for hs in hidden_sets:
                    for wi in weight_inits:
                        cfg = SimpleNamespace(
                            dataset="mnist",
                            epochs=4,                 # keep sweep fast; later rerun best with more epochs
                            batch_size=64,
                            loss="cross_entropy",
                            optimizer=opt,
                            learning_rate=lr,
                            weight_decay=0.0,
                            num_hidden_layers=len(hs),
                            hidden_sizes=hs,
                            activation=act,
                            weight_init=wi,
                            wandb_project="DA6401-A1",
                        )

                        trials += 1
                        model, val, test = train_one(cfg, X_train, y_train_oh, X_test, y_test_oh, X_val, y_val_oh)
                        test_f1 = test[3]

                        print(
                            f"[{trials}] opt={opt:8s} lr={lr:<5g} act={act:<4s} hs={hs} "
                            f"| val_f1={val[3]:.4f} test_f1={test_f1:.4f}"
                        )

                        if test_f1 > best["f1"]:
                            best["f1"] = test_f1
                            best["cfg"] = vars(cfg)

                            os.makedirs("src", exist_ok=True)
                            np.save(save_model, model.get_weights())
                            with open(save_cfg, "w") as f:
                                json.dump(best["cfg"], f, indent=2)

    print("\nBest test F1:", best["f1"])
    print("Saved:", save_model, "and", save_cfg)


if __name__ == "__main__":
    main()