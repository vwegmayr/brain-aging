import unittest
import numpy as np

import src.test_retest.numpy_utils as np_utils


class TestPredictionRobustness(unittest.TestCase):
    def test_equal_and_correct(self):
        true_labels = np.array([0, 0, 0, 1])
        Y = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 1]
        ])

        r = np_utils.equal_and_correct_pairs(Y, true_labels)
        self.assertEqual(r, 1.0)

        true_labels = np.array([0, 0, 0, 1])
        Y = np.array([
            [1, 1],
            [0, 0],
            [0, 0],
            [1, 1]
        ])

        r = np_utils.equal_and_correct_pairs(Y, true_labels)
        self.assertEqual(r, 0.75)

        true_labels = np.array([0, 0, 0, 1])
        Y = np.array([
            [1, 1],
            [0, 0],
            [0, 0],
            [0, 0]
        ])

        r = np_utils.equal_and_correct_pairs(Y, true_labels)
        self.assertEqual(r, 0.5)

        true_labels = np.array([0, 0, 0, 1])
        Y = np.array([
            [0, 1],
            [0, 0],
            [0, 1],
            [1, 0]
        ])

        r = np_utils.equal_and_correct_pairs(Y, true_labels)
        self.assertEqual(r, 0.25)


if __name__ == "__main__":
    unittest.main()
