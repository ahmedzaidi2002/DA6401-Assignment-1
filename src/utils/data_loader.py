import numpy as np
from keras.datasets import mnist, fashion_mnist


def load_data(dataset_name):
    """
    Loads MNIST or Fashion-MNIST dataset.

    Returns:
        (X_train, y_train), (X_test, y_test)
    """

    dataset_name = dataset_name.lower()

    if dataset_name == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    elif dataset_name == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    else:
        raise ValueError("Dataset must be 'mnist' or 'fashion_mnist'")

    # Flatten images (28x28 -> 784)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Normalize pixels
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    return (X_train, y_train), (X_test, y_test)


def one_hot_encode(y, num_classes=10):
    """
    Converts label vector to one-hot encoding.
    """

    encoded = np.zeros((len(y), num_classes), dtype=np.float32)
    encoded[np.arange(len(y)), y] = 1.0
    return encoded