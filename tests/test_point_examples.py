import unittest
from modules.models.example_loader import PointExamples

class TestPointExamples(unittest.TestCase):
    """Test some functionalities of the example loader."""

    @classmethod
    def setUpClass(cls):
        TRK = "data/iFOD2_skip100.trk"
        NII = "data/FODl4.nii.gz"
        PATH = "tests"
        cls.loader = PointExamples(
            NII,
            TRK,
            block_size=3,
            num_eval_examples=0,
            example_percent=0.25,
        )

    @classmethod
    def tearDownClass(cls):
        pass

    def test_example_percent(self):
        num_train_labels = len(self.loader.train_labels)
        num_eval_labels = len(self.loader.eval_labels)
        self.assertEqual(num_train_labels, 32328)
        self.assertEqual(num_eval_labels, 0)


if __name__ == '__main__':
    unittest.main()
