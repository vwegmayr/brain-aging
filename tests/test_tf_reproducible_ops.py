import unittest
import tensorflow as tf
import numpy as np


class TestTFReproducibility(unittest.TestCase):
    def setUp(self):
        self.session = tf.Session()

    def tearDown(self):
        self.session.close()

    def test_softmax(self):
        for i in range(10):
            np.random.seed(11)
            n_classes = 2
            n_samples = 10
            logits = np.random.rand(n_samples, n_classes)

            tf_logits = tf.placeholder(tf.float32, shape=[n_samples, n_classes])

            s = tf.nn.softmax(logits)

            r = self.session.run(s, {
                tf_logits: logits
            })
            print(r)

    def test_sparse_softmax_cross_entropy(self):
        for i in range(10):
            np.random.seed(11)
            n_classes = 2
            n_samples = 100
            logits = np.random.rand(n_samples, n_classes)
            labels = np.random.randint(0, n_classes, n_samples)
            labels = np.reshape(labels, (n_samples, 1))

            tf_logits = tf.placeholder(tf.float32, shape=[n_samples, n_classes])
            tf_labels = tf.placeholder(tf.float32, shape=[n_samples, 1])

            loss = tf.losses.sparse_softmax_cross_entropy(
                labels=labels,
                logits=logits
            )

            r = self.session.run(loss, {
                tf_labels: labels,
                tf_logits: logits
            })
            print(r)


if __name__ == "__main__":
    unittest.main()
