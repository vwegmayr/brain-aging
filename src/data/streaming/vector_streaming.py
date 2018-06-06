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

    def get_data_matrices(self, train=True):
        X = []
        Y = []
        target_key = self.config["target_label_key"]
        groups = [group for group in self.groups if group.is_train == train]

        for g in groups:
            paths = [self.get_file_path(fid) for fid in g.file_ids]
            vecs = [self.load_sample(p) for p in paths]
            vecs = np.array(vecs).ravel()
            X.append(vecs)

            # collect labels
            labs = [self.get_meta_info_by_key(fid, target_key)
                    for fid in g.file_ids]

            Y.append(np.array(labs).ravel())

        X = np.array(X)
        Y = np.array(Y)

        return X, Y
