import copy
import abc
import numpy as np
import tensorflow as tf

from src.data.mnist import read as mnist_read


class MnistStream(object):
    def __init__(self, stream_config):
        self.config = config = copy.deepcopy(stream_config)
        self.np_random = np.random.RandomState(seed=config["seed"])

    @abc.abstractmethod
    def get_input_fn(self, train):
        pass

    def dump_split(self, path):
        pass

    def dump_normalization(self, pathh):
        pass


class MnistPairStream(MnistStream):
    def get_input_fn(self, train):
        if train:
            # load training data
            test, retest, test_labels, retest_labels = \
                mnist_read.load_test_retest_two_labels(
                    self.config["data_path"],  # path to MNIST root folder
                    self.config["train_test_retest"],
                    self.config["train_size"],
                    True,
                    self.config["mix_pairs"]
                )

        else:
            # load test/retest data
            if "true_test_data" not in self.config:
                # Load sampled data
                test, retest, test_labels, retest_labels = \
                    mnist_read.load_test_retest_two_labels(
                        self.config["data_path"],  # path to MNIST root folder
                        self.config["test_test_retest"],
                        self.config["test_size"],
                        False,
                        self.config["mix_pairs"]
                    )
            else:
                # Load unmodified original test data
                print(">>>>>>>> Loading true test data")
                images, labels = mnist_read.load_mnist_test(
                    self.config["true_test_data"]
                )
                images = images[:self.config["test_size"]]
                labels = labels[:self.config["test_size"]]
                # Make pairs
                half = int(len(images) / 2)
                test = images[:half]
                retest = images[half:2*half]
                test_labels = labels[:half]
                retest_labels = labels[half:2*half]

        if self.config["shuffle"]:
            # shuffle data
            n = len(test_labels)
            idx = list(range(n))
            self.np_random.shuffle(idx)
            test = test[idx]
            retest = retest[idx]
            test_labels = test_labels[idx]
            retest_labels = retest_labels[idx]

        test_names = [str(i) + "_0" for i in range(len(test_labels))]
        retest_names = [str(i) + "_1" for i in range(len(test_labels))]
        features = {
            "X_0": test,
            "X_1": retest,
            "digit_0": test_labels,
            "digit_1": retest_labels,
            "file_name_0": test_names,
            "file_name_1": retest_names,
        }

        dataset = tf.data.Dataset.from_tensor_slices(
            (features, len(test_names) * [0])
        )
        dataset = dataset.batch(batch_size=self.config["batch_size"])

        def _input_fn():
            return dataset.make_one_shot_iterator().get_next()
        return _input_fn
