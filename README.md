# DA6401 Assignment 1 — NumPy Multi-Layer Perceptron

**GitHub:** [https://github.com/ahmedzaidi2002/DA6401-Assignment-1](https://github.com/ahmedzaidi2002/DA6401-Assignment-1)

**W&B Report:** [Assignment 1 Report](https://wandb.ai/cs25m009-indian-institute-of-technology-madras/Assignment_1/reports/Assignment-1-Report--VmlldzoxNjEwMTQyOQ?accessToken=pss9ahjqnloy4m4fx0itw6vi6a8p1b4k5i6hj58tnh20he4qr2dtvvk7r0ut2ggq)

---

## Overview

A fully configurable Multi-Layer Perceptron (MLP) built **from scratch using only NumPy**. The project implements the complete training pipeline — forward propagation, backpropagation, multiple optimizers, and various activation/loss functions — to classify the **MNIST** and **Fashion-MNIST** datasets.

No automatic differentiation frameworks (PyTorch, TensorFlow, JAX) are used. All gradients are derived and implemented analytically.

---

## Project Structure

```
.
├── README.md
├── requirements.txt
├── models/
│   └── .gitkeep
└── src/
    ├── __init__.py
    ├── best_model.npy          # Serialized best model weights
    ├── best_config.json        # Hyperparameters for the best model
    ├── train.py                # Training script with CLI
    ├── inference.py            # Inference script — loads model & reports metrics
    ├── ann/
    │   ├── __init__.py
    │   ├── activations.py      # Sigmoid, Tanh, ReLU, Softmax
    │   ├── neural_layer.py     # Fully-connected layer (forward + backward)
    │   ├── neural_network.py   # Model class — wires layers, loss, and optimizer
    │   ├── objective_functions.py  # MSE and Cross-Entropy loss
    │   └── optimizers.py       # SGD, Momentum, NAG, RMSProp
    └── utils/
        ├── __init__.py
        └── data_loader.py      # MNIST / Fashion-MNIST loader (multiple fallbacks)
```

---

## Features

### Activations

| Function | Forward                                     | Backward                |
| -------- | ------------------------------------------- | ----------------------- |
| Sigmoid  | Numerically stable split for pos/neg inputs | σ(z)(1 − σ(z))          |
| Tanh     | `np.tanh`                                   | 1 − tanh²(z)            |
| ReLU     | max(0, z)                                   | 1 if z > 0, else 0      |
| Softmax  | Shifted exp for numerical stability         | Vector-Jacobian product |

### Loss Functions

| Loss          | Description                                              |
| ------------- | -------------------------------------------------------- |
| Cross-Entropy | Softmax applied internally; gradient = (softmax − y) / m |
| MSE           | Mean Squared Error; gradient = 2(ŷ − y) / (m·C)          |

### Optimizers

| Optimizer | Description                                                       |
| --------- | ----------------------------------------------------------------- |
| SGD       | Vanilla stochastic gradient descent                               |
| Momentum  | SGD with exponential moving average of gradients (β = 0.9)        |
| NAG       | Nesterov Accelerated Gradient — look-ahead momentum               |
| RMSProp   | Adaptive learning rate using running average of squared gradients |

### Weight Initialization

- **Random:** Small Gaussian noise (σ = 0.01)
- **Xavier:** Uniform within ±√(6 / (fan_in + fan_out))

---

## Installation

```bash
git clone https://github.com/ahmedzaidi2002/DA6401-Assignment-1.git
cd DA6401-Assignment-1
pip install -r requirements.txt
```

### Dependencies

- `numpy`
- `scikit-learn` (train/val split)
- `matplotlib` (visualization)
- `wandb` (experiment tracking)
- `keras` or `tensorflow` (data loading only)

---

## Usage

### Training

```bash
python src/train.py \
  -d mnist \
  -e 20 \
  -b 64 \
  -l cross_entropy \
  -o rmsprop \
  -lr 0.001 \
  -wd 0.0001 \
  -nhl 3 \
  -sz 128 128 64 \
  -a relu \
  -w_i xavier
```

Every run saves the best epoch's weights to `src/best_model.npy` and its configuration to `src/best_config.json`.

### Inference

```bash
python src/inference.py --model_path src/best_model.npy --config_path src/best_config.json
```

Outputs Accuracy, Precision, Recall, and F1-score on the test set.

### Full CLI Arguments

| Flag   | Long Form         | Description                         | Default         |
| ------ | ----------------- | ----------------------------------- | --------------- |
| `-d`   | `--dataset`       | `mnist` or `fashion_mnist`          | `mnist`         |
| `-e`   | `--epochs`        | Number of training epochs           | `10`            |
| `-b`   | `--batch_size`    | Mini-batch size                     | `64`            |
| `-l`   | `--loss`          | `mse` or `cross_entropy`            | `cross_entropy` |
| `-o`   | `--optimizer`     | `sgd`, `momentum`, `nag`, `rmsprop` | `momentum`      |
| `-lr`  | `--learning_rate` | Initial learning rate               | `0.001`         |
| `-wd`  | `--weight_decay`  | L2 regularization strength          | `0.0`           |
| `-nhl` | `--num_layers`    | Number of hidden layers             | `2`             |
| `-sz`  | `--hidden_size`   | Neurons per hidden layer (list)     | `128 64`        |
| `-a`   | `--activation`    | `sigmoid`, `tanh`, `relu`           | `relu`          |
| `-w_i` | `--weight_init`   | `random` or `xavier`                | `xavier`        |
| `-wp`  | `--wandb_project` | W&B project name                    | `Assignment_1`  |

---

## Best Model Configuration

```json
{
  "dataset": "mnist",
  "epochs": 20,
  "batch_size": 64,
  "loss": "cross_entropy",
  "optimizer": "rmsprop",
  "learning_rate": 0.001,
  "weight_decay": 0.0001,
  "num_hidden_layers": 3,
  "hidden_sizes": [128, 128, 64],
  "activation": "relu",
  "weight_init": "xavier"
}
```

Architecture: **784 → 128 → 128 → 64 → 10**

---

## W&B Experiments Summary

The full experiment report is available [here](https://wandb.ai/cs25m009-indian-institute-of-technology-madras/Assignment_1/reports/Assignment-1-Report--VmlldzoxNjEwMTQyOQ?accessToken=pss9ahjqnloy4m4fx0itw6vi6a8p1b4k5i6hj58tnh20he4qr2dtvvk7r0ut2ggq). Key experiments include:

1. **Data Exploration** — Sample images from all 10 classes logged as a W&B Table.
2. **Hyperparameter Sweep** — 100+ runs varying optimizer, learning rate, batch size, architecture, and activation. Parallel Coordinates plots used to identify the most impactful hyperparameters.
3. **Optimizer Comparison** — Convergence rates of SGD, Momentum, NAG, and RMSProp on a fixed architecture (3 hidden layers, 128 neurons, ReLU).
4. **Vanishing Gradient Analysis** — Gradient norms for Sigmoid vs ReLU across different depths.
5. **Dead Neuron Investigation** — ReLU with high learning rate (0.1) showing activation distributions and dead neuron counts vs Tanh baseline.
6. **Loss Function Comparison** — MSE vs Cross-Entropy training curves on identical architectures.
7. **Global Performance Analysis** — Training vs Test accuracy overlay across all sweep runs to identify overfitting.
8. **Confusion Matrix & Error Analysis** — Per-class error breakdown for the best model.
9. **Weight Initialization Study** — Zero init vs Xavier init with gradient plots showing symmetry breaking.
10. **Fashion-MNIST Transfer** — Top 3 configurations from MNIST experiments applied to Fashion-MNIST.

---

## Author

**Ahmed Zaidi** — CS25M009, IIT Madras

DA6401: Introduction to Deep Learning, Assignment 1
