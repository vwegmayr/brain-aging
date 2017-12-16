import unittest

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D     # Import needed for 3D plotting
import numpy as np

from modules.models.base import ProbabilisticTracker as pt


class TestVMFSamplint(unittest.TestCase):

    def test_sample_unif_unit_circle(self):
        n_samples = 300
        samples = pt.sample_unif_unit_circle(n_samples)
        # plt.scatter(samples[:, 0], samples[:, 1])
        # plt.show()

    def test_sample_vMF(self):
        n_samples = 500
        mu = [[1,0,0]] * n_samples
        k = [4] * n_samples
        o = pt.sample_vMF(mu, k)
        print(o.shape)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_aspect('equal')
        print(o[:,0])
        print(o[:,1])
        print(o[:,2])
        ax.scatter(o[:, 0], o[:, 1], o[:, 2])
        # plt.show()

if __name__ == '__main__':
    unittest.main()
