"""THE TESTS IN THIS MODULE ARE STUBSself."""

import unittest

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # Import needed for 3D plotting

from modules.models.base import ProbabilisticTracker as pt


class TestVMFSamplint(unittest.TestCase):

    def test_sample_unif_unit_circle(self):
        n_samples = 300
        samples = pt.sample_unif_unit_circle(n_samples)
        # plt.scatter(samples[:, 0], samples[:, 1])
        # plt.show()

    def test_sample_vMF(self):
        n_samples = 500
        mu = [[1, 0, 0]] * n_samples
        k = [5] * n_samples
        samples = pt.sample_vMF(mu, k)
        print(samples)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # ax.set_aspect('equal')
        ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2])
        plt.show()

    def test_rotations(self):
        vectors = np.asarray([[1, 0, 0]] * 4)
        references = np.asarray([[0, 1, 0]] * 4)
        rot = pt._rotation_matrices(vectors, references)


if __name__ == '__main__':
    unittest.main()
