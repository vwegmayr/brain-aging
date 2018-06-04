import os
import json
import numpy as np
import importlib
import matplotlib.pyplot as plt

from modules.models.data_transform import DataTransformer

JSON_TYPE = '.json'
NUMPY_TYPE = '.npy'
FILE_TYPES = [JSON_TYPE, NUMPY_TYPE]


class RobustnessMeasureComputation(DataTransformer):
    def __init__(self, robustness_funcs, features_path, file_type,
                 streamer_collection, file_name_key):
        # Parse functions
        self.robustness_funcs = []
        for f in robustness_funcs:
            module_str = ".".join(f.split(".")[:-1])
            module = importlib.import_module(module_str)
            func = getattr(module, f.split(".")[-1])
            self.robustness_funcs.append(func)

        self.features_path = features_path
        # Initialize streamers
        _class = streamer_collection["class"]
        _params = streamer_collection["params"]
        self.streamer_collection = _class(**_params)
        self.streamers = self.streamer_collection.get_streamers()
        self.file_type = file_type
        self.file_name_key = file_name_key

    def get_file_name(self, file_id):
        streamer = self.streamers[0]
        return streamer.get_meta_info_by_key(file_id, self.file_name_key)

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

    def process_stream(self, streamer):
        # Steam batches (only one batch expected)
        batches = streamer.get_batches()
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

        return computation_dic

    def get_robustness_dic(self, computation_dic):
        feature_names = list(computation_dic.keys())
        some_feature = feature_names[0]
        robustness_names = list(computation_dic[some_feature].keys())
        robustness_names.remove('n_samples')
        robustness_to_vals = {}
        for r_name in robustness_names:
            values = []
            for f_name in feature_names:
                values.append(computation_dic[f_name][r_name])
            robustness_to_vals[r_name] = values

        return robustness_to_vals

    def generate_histo(self, values, labels, title, xlabel, ylabel, file_path):
        plt.figure(figsize=(10, 6))
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.hist(values, edgecolor='black', label=labels)
        plt.legend(loc=1, ncol=1)
        plt.tight_layout()
        plt.show()

    def generate_streamer_histogram(self, computation_dic, file_path):
        dic = self.get_robustness_dic(computation_dic)
        all_values = []
        labels = []
        for r_name in dic:
            labels.append(r_name)
            all_values.append(dic[r_name])

        self.generate_histo(
            values=all_values,
            labels=labels,
            title="Robustness histogram",
            xlabel="Robustness Value",
            ylabel="Count",
            file_path=file_path
        )

    def transform(self, X, y=None):
        """
        X and y are None. Data that is read is specified
        by 'self.streamer'.
        """
        out_path = os.path.join(self.save_path, "robustness_measures")
        os.mkdir(out_path)

        # Compute robustness for features and streamers
        streamer_to_comp = {}
        for streamer in self.streamer_collection.get_streamers():
            computation_dic = self.process_stream(streamer)
            file_name = streamer.name + "_" + "computations.json"
            # Dump computations
            with open(os.path.join(out_path, file_name), 'w') as f:
                json.dump(computation_dic, f, indent=2)

            histo_path = os.path.join(out_path, streamer.name + "_histo.pdf")
            # Generate histogram
            self.generate_streamer_histogram(computation_dic, histo_path)

            streamer_to_comp[streamer] = computation_dic

        # Compare robusntess of different streamers

        # Make pickable
        self.streamers = None
        self.streamer_collection = None
