import os
import unittest

import numpy as np
from sklearn.externals import joblib
from shutil import rmtree
import modules.models.utils as utils


class Test_make_data_sets(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        if not os.path.exists("tests/data"):
            os.mkdir("tests/data")
        self.paths = utils.make_rand_sets("tests/data")

    @classmethod
    def tearDownClass(self):
        rmtree("tests/data")

    def test_if_pkls_are_created(self):
        for mode in ["train", "test"]:
            for _, file in self.paths[mode].items():
                self.assertTrue(os.path.exists(file))

    def test_train_shapes(self):
        X = joblib.load(self.paths["train"]["X"])
        self.assertEqual(X["blocks"].shape, (100, 3, 3, 3, 15))
        self.assertEqual(X["incoming"].shape, (100, 9))

        y = joblib.load(self.paths["train"]["y"])
        self.assertEqual(y.shape, (100, 3))

    def test_test_shapes(self):
        X = joblib.load(self.paths["test"]["X"])
        self.assertEqual(X["dwi"].shape, (10, 10, 10, 15))
        self.assertEqual(X["mask"].shape, (10, 10, 10))


class Test_aff_to_rot(unittest.TestCase):

    def test_if_rotation_is_correct(self):
        # 90 deg rotation
        aff = np.asarray(
            [[0, -2, 0, 10],
             [1,  0, 0, 20],
             [0,  0, 3, 30],
             [0,  0, 0, 1]
            ]
        )
        rot = utils.aff_to_rot(aff)

        self.assertEqual(rot.tolist(), [[0, -1, 0], [1, 0, 0], [0, 0, 1]])


if __name__ == '__main__':
    unittest.main()
