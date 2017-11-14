import numpy as np

from abc import ABC, abstractmethod
from tensorflow.examples.tutorials.mnist import input_data

class Data(ABC):
    """docstring for Data"""
    def __init__(self):
        super(Data, self).__init__()

    @abstractmethod
    def get_batch(self, batch_size, type):
        pass

    @abstractmethod
    def input_shape(self):
        pass

    @abstractmethod
    def output_shape(self):
        pass

    @abstractmethod
    def num_samples(self, type):
        pass


class MNIST(Data):
    """docstring for MNIST"""

    def __init__(self, path="data/mnist"):
        super(MNIST, self).__init__()
        self._data_set = input_data.read_data_sets(path, one_hot=True)

    def get_batch(self, batch_size, type):
        X, y = getattr(self._data_set, type).next_batch(batch_size)          
        return X.reshape([batch_size] + self.input_shape()), y

    def input_shape(self):
        return [28, 28, 1]

    def output_shape(self):
        return [10]

    def num_samples(self, type):
        if type in ["train", "validation", "test"]:
            return getattr(self._data_set, type)._num_examples
        else:
            return 0


class DataLoader(Data):
    """docstring for DataLoader"""
    def __init__(self, X, y=None):
        super(DataLoader, self).__init__()
        self.X = X
        self.y = y
        self.sample_yield = 0
        self.random_indices = np.random.permutation(self.num_samples("train"))

    def get_batch(self, batch_size, type):       
        if self.sample_yield <= self.num_samples("train") - batch_size:
            idx = self.random_indices[self.sample_yield : self.sample_yield + batch_size]
            X = self.X[idx, :]
            y = self.y[idx, :]
            self.sample_yield += batch_size
            return X, y
        else:
            self.sample_yield = 0
            self.random_indices = np.random.permutation(self.num_samples("train"))
            return self.get_batch(batch_size, type)

    def input_shape(self):
        return list(self.X.shape[1:])

    def output_shape(self):
        return list(self.y.shape[1:])

    def num_samples(self, type):
        if type == "train":
            return self.X.shape[0]
        else:
            return 0

    def random_sequence(self):
        pass

    def X(self):
        return self.X