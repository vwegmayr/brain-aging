import os
import unittest

import numpy as np
from sklearn.externals import joblib

import modules.models.utils as utils


class Test_Nii_Trk_To_Pkl_Conversion(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        TRK = "data/iFOD2_skip100.trk"
        NII = "data/FODl4.nii.gz"
        PATH = "tests"

        utils.convert_nii_and_trk_to_pkl(
            NII,
            TRK,
            block_size=3,
            pkl_path=PATH,
            samples_percent=0.3)

        if os.path.exists("tests_X.pkl"):
            cls.X = joblib.load("tests_X.pkl")
        else:
            cls.X = None
        if os.path.exists("tests_y.pkl"):
            cls.y = joblib.load("tests_y.pkl")
        else:
            cls.y = None

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("tests_X.pkl"):
            os.remove("tests_X.pkl")
        if os.path.exists("tests_y.pkl"):
            os.remove("tests_y.pkl")

    def test_if_pkls_are_created(self):
        self.assertIsNotNone(self.X)
        self.assertIsNotNone(self.y)

    def test_X_blocks_shape(self):
        self.assertEqual(self.X["blocks"].shape, (38794, 3, 3, 3, 15))
        self.assertEqual(self.y.shape, (38794, 3))


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
