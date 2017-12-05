import os
import unittest
import modules.models.utils as utils
from sklearn.externals import joblib

class Test_Nii_Trk_To_Pkl_Conversion(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        TRK = "data/iFOD2.trk"
        NII = "data/FODl4.nii.gz"
        PATH = "tests"

        utils.convert_nii_and_trk_to_npy(
            NII,
            TRK,
            block_size=3,
            path=PATH,
            n_samples=100)

        if os.path.exists("tests/X.pkl"):
            cls.X = joblib.load("tests/X.pkl")
        else:
            cls.X = None
        if os.path.exists("tests/y.pkl"):
            cls.y = joblib.load("tests/y.pkl")
        else:
            cls.y = None
    
    @classmethod
    def tearDownClass(cls):
        if os.path.exists("tests/X.pkl"):
            os.remove("tests/X.pkl")
        if os.path.exists("tests/y.pkl"):
            os.remove("tests/y.pkl")


    def test_if_pkls_are_created(self):
        self.assertIsNotNone(self.X)
        self.assertIsNotNone(self.y)


    def test_X_blocks_shape(self):
        self.assertEqual(self.X["blocks"].shape, (100, 3, 3, 3, 15))
        self.assertEqual(self.y.shape, (100, 3))



if __name__ == '__main__':
    unittest.main()
