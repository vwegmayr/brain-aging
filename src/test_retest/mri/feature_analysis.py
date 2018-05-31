import os
import json
import numpy as np
import importlib

from modules.models.data_transform import DataTransformer


class RobustnessMeasureComputation(DataTransformer):
    def __init__(self, robustness_funcs, features_path, streamer):
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

    def features_exist(self, im_label):
        return os.path.isfile(os.path.join(self.features_path, str(im_label) + ".json"))

    def load_features(self, im_label):
        with open(os.path.join(self.features_path, str(im_label) + ".json")) \
         as f:
            features_dic = json.load(f)

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
            im_label_1 = self.streamer.get_image_label(ids[0])
            im_label_2 = self.streamer.get_image_label(ids[1])
            if self.features_exist(im_label_1) and \
                    self.features_exist(im_label_2):
                print("features not found for {} and {}"
                      .format(im_label_1, im_label_2))
                f1 = self.load_features(im_label_1)
                f2 = self.load_features(im_label_2)
                features.append((f1, f2))

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
