import unittest
import tensorflow as tf
import numpy as np
from src.test_retest.regularizer import \
    js_divergence, batch_divergence

from src.test_retest import numpy_utils


class TestRegularizers(unittest.TestCase):
    def setUp(self):
        self.sess = tf.Session()

    def tearDown(self):
        self.sess.close()

    def test_js_divergence(self):
        d = 20
        n = 10000
        A = np.random.rand(n, 2 * d)

        x1 = tf.placeholder(
            dtype=tf.float64,
            shape=[n, d],
            name="x1"
        )

        x2 = tf.placeholder(
            dtype=tf.float64,
            shape=[n, d],
            name="x2"
        )

        y = batch_divergence(
            x1,
            x2,
            d,
            js_divergence
        )

        res = self.sess.run(
            y,
            {
                x1: A[:, :d],
                x2: A[:, d:]
            }
        )

        comp = numpy_utils.batch_divergence(
            A[:, :d],
            A[:, d:],
            numpy_utils.js_divergence
        )

        self.assertTrue(np.allclose(res, comp))


if __name__ == "__main__":
    unittest.main()
