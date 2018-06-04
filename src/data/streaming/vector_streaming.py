import numpy as np

from .base import FileStream
from .mri_streaming import MRISingleStream


class FeatureVectorStream(MRISingleStream):
    def load_sample(self, file_path):
        return np.load(file_path)

    def make_train_test_split(self):
        # train files
        ds_train = self.get_data_source_by_name("train")
        train_paths = set(ds_train.get_file_paths())

        for group in self.groups:
            fid = group.file_ids[0]
            p = self.get_file_path(fid)
            group.is_train = (p in train_paths)
