import os
import unittest
import modules.models.utils as utils

class TestUtil(unittest.TestCase):

    def test_loader(self):
        trk_file = "data/iFOD2.trk"
        nii_file = "data/FODl4.nii.gz"
        path = "data"
        utils.convert_nii_and_trk_to_npy(nii_file, trk_file, block_size=3, path=path)

        self.assertTrue(os.path.exists("data/X.npy"))
        self.assertTrue(os.path.exists("data/y.npy"))

if __name__ == '__main__':
    unittest.main()
