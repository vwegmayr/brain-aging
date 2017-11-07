from abc import ABC, abstractmethod
from tensorflow.examples.tutorials.mnist import input_data

class Data(ABC):
    """docstring for Data"""
    def __init__(self):
        super(Data, self).__init__()

    @abstractmethod
    def get_train_batch(self, batch_size):
        pass

    @abstractmethod
    def get_eval_batch(self, batch_size):
        pass

    @abstractmethod
    def input_shape(self):
        pass

    @abstractmethod
    def output_shape(self):
        pass

    @abstractmethod
    def num_train_samples(self):
        pass

    @abstractmethod
    def num_test_samples(self):
        pass


class MNIST(Data):
    """docstring for MNIST"""

    def __init__(self, path="data/mnist"):
        super(MNIST, self).__init__()
        self._data_set = input_data.read_data_sets(path, one_hot=True)

    def get_train_batch(self, batch_size):
        X, y = self._data_set.train.next_batch(batch_size)          
        return X.reshape([batch_size] + self.input_shape()), y

    def get_eval_batch(self, batch_size):
        X, y = self._data_set.test.next_batch(batch_size)          
        return X.reshape([batch_size] + self.input_shape()), y

    def input_shape(self):
        return [28, 28, 1]

    def output_shape(self):
        return [10]

    def num_train_samples(self):
        return self._data_set.train._num_examples

    def num_test_samples(self):
        return self._data_set.test._num_examples


class FromFile(Data):
    """docstring for FromFile"""
    def __init__(self, X, y):
        super(FromFile, self).__init__()
        self.X = X
        self.y = y

    def get_train_batch(self, batch_size):
        pass

    def get_eval_batch(self, batch_size):
        pass

    def input_shape(self):
        return self.X.shape[0]

    def output_shape(self):
        return [10]

    def num_train_samples(self):
        pass
        
    def num_test_samples(self):
        pass        
        