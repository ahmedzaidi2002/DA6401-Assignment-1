import os
import gzip
import struct
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

# Keras stores mnist as a single .npz file at this path
_KERAS_NPZ = {
    "mnist":         os.path.expanduser("~/.keras/datasets/mnist.npz"),
    "fashion_mnist": os.path.expanduser("~/.keras/datasets/fashion-mnist.npz"),
}


def _load_images(path):
    with gzip.open(path, "rb") as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float32) / 255.0


def _load_labels(path):
    with gzip.open(path, "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def _try_keras_npz(dataset_name):
    """Try loading from keras pre-cached .npz file."""
    npz_path = _KERAS_NPZ.get(dataset_name)
    if npz_path and os.path.exists(npz_path):
        try:
            d = np.load(npz_path)
            X_tr = d["x_train"].reshape(len(d["x_train"]), -1).astype(np.float32) / 255.0
            X_te = d["x_test"].reshape(len(d["x_test"]), -1).astype(np.float32) / 255.0
            return (X_tr, d["y_train"]), (X_te, d["y_test"])
        except Exception:
            pass
    return None


def _try_tensorflow(dataset_name):
    try:
        from tensorflow.keras.datasets import mnist, fashion_mnist
        ds = mnist if dataset_name == "mnist" else fashion_mnist
        (X_tr, y_tr), (X_te, y_te) = ds.load_data()
        X_tr = X_tr.reshape(len(X_tr), -1).astype(np.float32) / 255.0
        X_te = X_te.reshape(len(X_te), -1).astype(np.float32) / 255.0
        return (X_tr, y_tr), (X_te, y_te)
    except Exception:
        return None


def _try_keras(dataset_name):
    try:
        import keras
        ds = keras.datasets.mnist if dataset_name == "mnist" else keras.datasets.fashion_mnist
        (X_tr, y_tr), (X_te, y_te) = ds.load_data()
        X_tr = X_tr.reshape(len(X_tr), -1).astype(np.float32) / 255.0
        X_te = X_te.reshape(len(X_te), -1).astype(np.float32) / 255.0
        return (X_tr, y_tr), (X_te, y_te)
    except Exception:
        return None


def _try_download(dataset_name):
    try:
        import urllib.request
        urls = _URLS[dataset_name]
        cache = os.path.join(_CACHE_DIR, dataset_name)
        os.makedirs(cache, exist_ok=True)

        files = {}
        for key, url in urls.items():
            dest = os.path.join(cache, key + ".gz")
            if not os.path.exists(dest):
                urllib.request.urlretrieve(url, dest)
            files[key] = dest

        X_train = _load_images(files["train_images"])
        y_train = _load_labels(files["train_labels"])
        X_test  = _load_images(files["test_images"])
        y_test  = _load_labels(files["test_labels"])
        return (X_train, y_train), (X_test, y_test)
    except Exception:
        return None


def load_data(dataset_name):
    """
    Loads MNIST or Fashion-MNIST. Tries multiple sources in order:
    1. Keras pre-cached .npz
    2. tensorflow.keras
    3. keras standalone
    4. Direct urllib download
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in _URLS:
        raise ValueError("Dataset must be 'mnist' or 'fashion_mnist'")

    for loader in [_try_keras_npz, _try_tensorflow, _try_keras, _try_download]:
        result = loader(dataset_name)
        if result is not None:
            return result

    raise RuntimeError(
        f"Could not load '{dataset_name}'. All loading methods failed. "
        "Ensure tensorflow, keras, or internet access is available."
    )


def one_hot_encode(y, num_classes=10):
    """Converts label vector to one-hot encoding."""
    encoded = np.zeros((len(y), num_classes), dtype=np.float32)
    encoded[np.arange(len(y)), y] = 1.0
    return encoded