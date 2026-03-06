import os
import gzip
import struct
import urllib.request
import numpy as np

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".mnist_cache")

_URLS = {
    "mnist": {
        "train_images": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz",
        "train_labels": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz",
        "test_images":  "https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz",
        "test_labels":  "https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz",
    },
    "fashion_mnist": {
        "train_images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
        "train_labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
        "test_images":  "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
        "test_labels":  "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
    },
}


def _download(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if not os.path.exists(dest_path):
        urllib.request.urlretrieve(url, dest_path)


def _load_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float32) / 255.0


def _load_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def _try_keras(dataset_name):
    """Try keras/tensorflow as a fast path if available."""
    try:
        from tensorflow.keras.datasets import mnist, fashion_mnist
        ds = mnist if dataset_name == "mnist" else fashion_mnist
        (X_tr, y_tr), (X_te, y_te) = ds.load_data()
        X_tr = X_tr.reshape(len(X_tr), -1).astype(np.float32) / 255.0
        X_te = X_te.reshape(len(X_te), -1).astype(np.float32) / 255.0
        return (X_tr, y_tr), (X_te, y_te)
    except Exception:
        pass
    try:
        import keras
        ds = keras.datasets.mnist if dataset_name == "mnist" else keras.datasets.fashion_mnist
        (X_tr, y_tr), (X_te, y_te) = ds.load_data()
        X_tr = X_tr.reshape(len(X_tr), -1).astype(np.float32) / 255.0
        X_te = X_te.reshape(len(X_te), -1).astype(np.float32) / 255.0
        return (X_tr, y_tr), (X_te, y_te)
    except Exception:
        pass
    return None


def load_data(dataset_name):
    """
    Loads MNIST or Fashion-MNIST dataset.
    Falls back to direct download if keras/tensorflow unavailable.

    Returns:
        (X_train, y_train), (X_test, y_test)
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in _URLS:
        raise ValueError("Dataset must be 'mnist' or 'fashion_mnist'")

    # Fast path: try keras first
    result = _try_keras(dataset_name)
    if result is not None:
        return result

    # Fallback: download raw IDX files directly
    urls = _URLS[dataset_name]
    cache = os.path.join(_CACHE_DIR, dataset_name)

    files = {}
    for key, url in urls.items():
        dest = os.path.join(cache, key + ".gz")
        _download(url, dest)
        files[key] = dest

    X_train = _load_images(files["train_images"])
    y_train = _load_labels(files["train_labels"])
    X_test  = _load_images(files["test_images"])
    y_test  = _load_labels(files["test_labels"])

    return (X_train, y_train), (X_test, y_test)


def one_hot_encode(y, num_classes=10):
    """Converts label vector to one-hot encoding."""
    encoded = np.zeros((len(y), num_classes), dtype=np.float32)
    encoded[np.arange(len(y)), y] = 1.0
    return encoded