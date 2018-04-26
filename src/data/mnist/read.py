import gzip
import os
import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    x_train, y_train = load_mnist_training("MNIST-data")
    print(x_train.shape)
    print(y_train.shape)

    print(y_train[100])
    plt.imshow(x_train[100], cmap='gray')
    plt.show()
