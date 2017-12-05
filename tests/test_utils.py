import os
import unittest
import modules.models.utils as utils

class TestUtil(unittest.TestCase):

    def test_convert_nii_and_trk_to_npy(self):
        trk_file = "data/iFOD2.trk"
        nii_file = "data/FODl4.nii.gz"

        path = "tests"
        utils.convert_nii_and_trk_to_npy(nii_file, trk_file, block_size=3, path=path, n_samples=100)
        self.assertTrue(os.path.exists("tests/X.npy"))
        self.assertTrue(os.path.exists("tests/y.npy"))
        os.remove("tests/X.npy")
        os.remove("tests/y.npy")


if __name__ == '__main__':
    unittest.main()
