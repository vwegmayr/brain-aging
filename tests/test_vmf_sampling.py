import unittest

import matplotlib.pyplot as plt
import numpy as np

from modules.models.base import ProbabilisticTracker as pt


class TestVMFSamplint(unittest.TestCase):

    def test_sample_unif_unit_circle(self):
        n_samples = 300
        samples = pt.sample_unif_unit_circle(n_samples)
        plt.scatter(samples[:, 0], samples[:, 1])
        # plt.show()

    def test_sample_vMF(self):
        n_samples = 50
        mu = [[1,0,0]] * n_samples
        k = [0.5] * n_samples
        print("Sampling")
        print(pt.sample_vMF(mu, k))


if __name__ == '__main__':
    unittest.main()
