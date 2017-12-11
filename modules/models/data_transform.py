from sklearn.base import BaseEstimator, TransformerMixin
from modules.models.utils import make_train_set

class TrainDataTransformer(BaseEstimator, TransformerMixin):
    """Convert dwi.nii and fiber.trk into a pickle.pkl

    This is a wrapper for utils.make_train_set, to enable
    data tracking with sumatra.
    
    """
    def __init__(
        self,
        dwi_file=None,
        trk_file=None,
        block_size=3,
        samples_percent=1.0,
        n_samples=None,
        min_fiber_length=0,
        n_incoming=1):

        super(TransformTrainData, self).__init__()

        self.dwi_file = dwi_file
        self.trk_file = trk_file
        self.block_size = block_size
        self.samples_percent = samples_percent
        self.n_samples = n_samples
        self.min_fiber_length = min_fiber_length
        self.n_incoming = n_incoming


    def fit(self, X=None, y=None):
        pass

    def transform(self, X=None, y=None):
        make_train_set(
            dwi_file = self.dwi_file,
            trk_file = self.trk_file,
            save_path = self.save_path,
            block_size = self.block_size,
            samples_percent = self.samples_percent,
            n_samples = self.n_samples,
            min_fiber_length = self.min_fiber_length,
            n_incoming = self.n_incoming
        )

    def set_save_path(self, save_path)
        self.save_path = save_path