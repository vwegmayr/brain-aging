import os
import json
import numpy as np
import importlib

from modules.models.data_transform import DataTransformer

JSON_TYPE = '.json'
NUMPY_TYPE = '.npy'
FILE_TYPES = [JSON_TYPE, NUMPY_TYPE]


class RobustnessMeasureComputation(DataTransformer):
    def __init__(self, robustness_funcs, features_path, file_type,
                 streamer, file_name_key):
        # Parse functions
        self.robustness_funcs = []
        for f in robustness_funcs:
            module_str = ".".join(f.split(".")[:-1])
            module = importlib.import_module(module_str)
            func = getattr(module, f.split(".")[-1])
            self.robustness_funcs.append(func)

        self.features_path = features_path
        # Parse streamer
        _class = streamer["class"]
        self.streamer = _class(**streamer["params"])

        self.file_type = file_type
        self.file_name_key = file_name_key

    def get_file_name(self, file_id):
        return self.streamer.get_meta_info_by_key[self.file_name_key]

    def construct_file_path(self, file_name):
        return os.path.join(self.features_path, file_name + self.file_type)

    def features_exist(self, file_name):
        p = self.construct_file_path(file_name)
        return os.path.isfile(p)

    def load_features(self, file_name):
        """
        Return:
            - features_dic: dictionary mapping feature names
              to their value
        """
        assert self.file_type in FILE_TYPES
        p = self.construct_file_path(file_name)
        with open(p) as f:
            if self.file_type == JSON_TYPE:
                features_dic = json.load(f)
            elif self.file_type == NUMPY_TYPE:
                features_vec = np.load(p)
                features_dic = {
                    str(i): val
                    for i, val in enumerate(features_vec)
                }

        return features_dic

    def transform(self, X, y=None):
        """
        X and y are None. Data that is read is specified
        by 'self.streamer'.
        """
        out_path = os.path.join(self.save_path, "robustness_measures")
        os.mkdir(out_path)
        # Steam batches (only one batch expected)
        batches = self.streamer.get_batches()
        assert len(batches) == 1
        batch = batches[0]

        features = []
        for group in batch:
            ids = group.get_file_ids()
            assert len(ids) == 2  # test-retest features
            # Read features for this group
            file_name_1 = self.get_file_name(ids[0])
            file_name_2 = self.get_file_name(ids[1])
            if self.features_exist(file_name_1) and \
                    self.features_exist(file_name_2):
                f1 = self.load_features(file_name_1)
                f2 = self.load_features(file_name_2)
                features.append((f1, f2))
            else:
                print("features not found for {} and {}"
                      .format(file_name_1, file_name_2))

        # Compute robustness measure using different features
        feature_names = features[0][0].keys()
        computation_dic = {}

        for name in feature_names:
            Y = []
            feature_dic = {}
            computation_dic[name] = feature_dic
            for f in features:
                Y.append([f[0][name], f[1][name]])

            feature_dic["n_samples"] = len(Y)
            Y = np.array(Y)
            for func in self.robustness_funcs:
                r = func(Y)
                feature_dic[func.__name__] = r

            print("Computed robustness for feature {}".format(name))

        self.streamer = None
        # Dump computations
        with open(os.path.join(out_path, "computations.json"), 'w') as f:
            json.dump(computation_dic, f, indent=2)
