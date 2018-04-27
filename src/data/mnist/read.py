import gzip
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import TransformerMixin


IMAGE_SIZE = 28


def read_gzip_images(file_path, n_images):
    """
    Assumes data from LeCun.
    Args:
        - file_path: path to gz file containing images
        - n_images: number of images to read

    Return:
        - data: numpy array of size (n_images, image_height, image_width)
          containg images
    """
    with gzip.open(file_path) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * n_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(n_images, IMAGE_SIZE, IMAGE_SIZE)

    return data


def read_gzip_labels(file_path, n_images):
    """
    Assumes data from LeCun.
    Args:
        - file_path: path to gz file containing images
        - n_images: number of images to read

    Return:
        - labels: numpy array containing image labels
    """
    with gzip.open(file_path) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * n_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

    return labels


def load_training_labels(folder_path):
    labels = read_gzip_labels(
        os.path.join(folder_path, "train-labels-idx1-ubyte.gz"),
        60000
    )

    return labels


def load_test_labels(folder_path):
    labels = read_gzip_labels(
        os.path.join(folder_path, "t10k-labels-idx1-ubyte.gz"),
        60000
    )

    return labels


def load_mnist_training(folder_path):
    """
    Assumes data from LeCun.
    Arg:
        - folder_path: path to folder containing data

    Returns:
        - images: numpy array containing images
        - labels: numpy array containing labels
    """
    # load images
    images = read_gzip_images(
        os.path.join(folder_path, "train-images-idx3-ubyte.gz"),
        60000
    )

    # load labels
    labels = read_gzip_labels(
        os.path.join(folder_path, "train-labels-idx1-ubyte.gz"),
        60000
    )

    return images, labels


def load_mnist_test(folder_path):
    """
    Assumes data from LeCun.
    Arg:
        - folder_path: path to folder containing data

    Returns:
        - images: numpy array containing images
        - labels: numpy array containing labels
    """
    # load images
    images = read_gzip_images(
        os.path.join(folder_path, "t10k-images-idx3-ubyte.gz"),
        10000
    )

    # load labels
    labels = read_gzip_labels(
        os.path.join(folder_path, "t10k-labels-idx1-ubyte.gz"),
        10000
    )

    return images, labels


def load_test_retest(data_path, test_rest_path, n_samples, train):
    # load labels
    if train:
        labels = load_training_labels(data_path)
    else:
        labels = load_test_labels(data_path)

    with open(test_rest_path, "rb") as f:
        X = np.load(f)

    return X[0, :n_samples, :, :], X[1, :n_samples, :, :], labels[:n_samples]


def sample_test_retest(n_pairs, images):
    # sample binary images using intensities as bernoulli images
    test = images
    retest = np.copy(images)

    for i in range(n_pairs):
        maxi = np.max(test[i, :, :])
        for j in range(IMAGE_SIZE):
            for k in range(IMAGE_SIZE):
                p = test[i, j, k] / maxi
                s1, s2 = np.random.binomial(1, p, 2)
                test[i, j, k] = s1
                retest[i, j, k] = s2

    return test, retest


def sample_test_retest_training(folder_path, n_pairs, seed):
    # Make sampling reproducible
    np.random.seed(seed)

    images, labels = load_mnist_training(folder_path)
    n = len(images)

    # sample source images
    idx = np.random.randint(0, n, min(n_pairs, n))
    images = images[idx]
    labels = labels[idx]

    test, retest = sample_test_retest(n_pairs, images)

    return test, retest, labels


def sample_test_retest_test(folder_path, n_pairs, seed):
    # Make sampling reproducible
    np.random.seed(seed)

    images, labels = load_mnist_test(folder_path)
    n = len(images)

    # sample source images
    idx = np.random.randint(0, n, min(n_pairs, n))
    images = images[idx]
    labels = labels[idx]

    test, retest = sample_test_retest(n_pairs, images)

    return test, retest, labels


class MnistSampler(TransformerMixin):
    def __init__(self, np_random_seed, data_path, train_data=True):
        self.np_random_seed = np_random_seed
        self.data_path = data_path
        self.train_data = train_data

    def fit(self, X):
        return self

    def transform(self, X, y=None):
        if self.train_data:
            X, _ = load_mnist_training(self.data_path)
        else:
            X, _ = load_mnist_test(self.data_path)

        n = len(X)
        sampled = sample_test_retest(n, X)

        return sampled


if __name__ == "__main__":
    x_train, y_train = load_mnist_training("MNIST-data")
    print(x_train.shape)
    print(y_train.shape)

    print(y_train[100])
    plt.imshow(x_train[100], cmap='gray')
    plt.show()
