from hashlib import sha1
from sklearn.base import BaseEstimator, TransformerMixin
from modules.models.utils import make_train_set, make_test_set


class DataTransformer(TransformerMixin):
    """docstring for DataTransformer"""
    def __init__(self):
        super(DataTransformer, self).__init__()

    def set_save_path(self, save_path):
        self.save_path = save_path


class TrainDataTransformer(DataTransformer):
    """Convert dwi.nii and fiber.trk into a pickle.pkl

    This is a wrapper for utils.make_train_set, to enable
    data tracking with sumatra.
    
    """
    def __init__(
        self,
        block_size=3,
        samples_percent=1.0,
        n_samples=None,
        min_fiber_length=0,
        n_incoming=1):

        super(TrainDataTransformer, self).__init__()

        self.block_size = block_size
        self.samples_percent = samples_percent
        self.n_samples = n_samples
        self.min_fiber_length = min_fiber_length
        self.n_incoming = n_incoming

    def transform(self, X, y=None):

        assert isinstance(X, list)
        assert len(X) == 2

        if "trk" in X[0] and "nii" in X[1]:
            X = X[::-1]

        make_train_set(
            dwi_file = X[0],
            trk_file = X[1],
            save_path = self.save_path,
            block_size = self.block_size,
            samples_percent = self.samples_percent,
            n_samples = self.n_samples,
            min_fiber_length = self.min_fiber_length,
            n_incoming = self.n_incoming
        )


class TestDataTransformer(DataTransformer):
    """docstring for TestDataTransformer"""
    def __init__(self):

        super(TestDataTransformer, self).__init__()

    def transform(self, X, y=None):
        
        assert isinstance(X, list)
        assert len(X) == 2

        if "mask" in X[0]:
            X = X[::-1]

        make_test_set(
            dwi_file=X[0],
            mask_file=X[1],
            save_path=self.save_path)