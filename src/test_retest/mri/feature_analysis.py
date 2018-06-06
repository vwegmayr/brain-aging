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
                 streamer_collection, file_name_key, output_dir):
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
        self.output_dir = output_dir

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
        """
        Arg:
            - computation_dic: dictionary mapping a feature name
              to a dictionary mapping to robustness measures to their
              values

        Return:
            - robustness_to_vals: dictionary mapping a robustness measure
              name to a list containing the robustness values of all the
              features for that robustness measure
        """
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
        """
        Generates a histogram for the given values.
        Args:
            - values: a list of value lists (comparing multiple series), or
              just a list of values
            - labels: list of strings, labels for the plots legend
            - title: title for the plot
            - xlabel: label for x-axis
            - ylabel: label for y-axis
            - file_path: path for plot figure
        """
        plt.clf()
        plt.figure(figsize=(10, 6))
        fs = 16
        plt.title(title, fontsize=fs)
        plt.xlabel(xlabel, fontsize=fs)
        plt.ylabel(ylabel, fontsize=fs)
        plt.hist(values, edgecolor='black', label=labels)
        plt.legend(loc=1, ncol=1, fontsize=fs-2)
        plt.tight_layout()
        plt.savefig(file_path)

    def generate_streamer_histogram(self, computation_dic, file_path):
        """
        Generate a histogram for the robustness computations of a single
        streamer.
        Args:
            - computation_dic: dictionary mapping a feature name
              to a dictionary mapping to robustness measures to their
              values
            - file_path: path for plot figure
        """
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

    def compare_computations(self, comps, names, out_path, same_patient=True):
        """
        Compare two computation dictionaries. A histogram is generated
        for every robustness measure comparing the two dictionaries.

        Args:
            - comp1: first computation dictionary
            - comp2: second computation dicitonary

        """
        dics = [self.get_robustness_dic(comp) for comp in comps]
        labels = names
        s = "same_patient_samplers"
        if not same_patient:
            s = "different_patient_samplers"
        for r_name in dics[0]:
            values = [dic[r_name] for dic in dics]
            file_name = "comparison_{}_{}.pdf".format(r_name, s)
            file_path = os.path.join(out_path, file_name)
            self.generate_histo(
                values=values,
                labels=labels,
                title="Histogram comparison",
                xlabel="Robustness Value {}".format(r_name),
                ylabel="Count",
                file_path=file_path
            )

    def get_feature_names(self, comp_dic):
        feature_names = list(comp_dic.keys())
        return feature_names

    def get_robustness_names(self, comp_dic):
        feature_names = self.get_feature_names(comp_dic)
        robustness_names = list(comp_dic[feature_names[0]].keys())
        robustness_names.remove("n_samples")

        return robustness_names

    def robustness_ratio(self, streamer_to_comp):
        """
        Arg:
            - streamer_to_comp: maps streamer object to
              computation dictionary
        """
        out_path = os.path.join(self.output_path, "predictive_power")
        os.makedirs(out_path)

        comp_dic = list(streamer_to_comp.values())[0]
        feature_names = self.get_feature_names(comp_dic)
        robustness_names = self.get_robustness_names(comp_dic)

        streamers = self.streamer_collection.get_different_patient_streamers()
        similar_streamers = [s for s in streamers if
                             s.get_diagnoses()[0] == s.get_diagnoses()[1]]
        dissimilar_streamers = [s for s in streamers if
                                s.get_diagnoses()[0] != s.get_diagnoses()[1]]

        similar_comps = [streamer_to_comp[s] for s in similar_streamers]
        dissimilar_comps = [streamer_to_comp[s] for s in dissimilar_streamers]
        similar_names = [s.name for s in similar_streamers]
        dissimilar_names = [s.name for s in dissimilar_streamers]

        robustness_to_feature_to_dic = {}
        for f_name in feature_names:
            for r_name in robustness_names:
                similar_scores = [abs(comp[f_name][r_name])
                                  for comp in similar_comps]
                dissimilar_scores = [abs(comp[f_name][r_name])
                                     for comp in dissimilar_comps]

                sim_mean = np.mean(similar_scores)
                dissim_mean = np.mean(dissimilar_scores)
                dic = {
                    'sim_names': similar_names,
                    'dissim_names': dissimilar_names,
                    'sim_scores': similar_scores,
                    'dissim_scores': dissimilar_scores,
                    'sim_mean': np.mean(similar_scores),
                    'sim_std': np.std(similar_scores),
                    'dissim_mean': np.mean(dissimilar_scores),
                    'dissim_std': np.std(dissimilar_scores),
                    'pred_score': sim_mean / dissim_mean,
                }

                if r_name not in robustness_to_feature_to_dic:
                    robustness_to_feature_to_dic[r_name] = {}
                robustness_to_feature_to_dic[r_name][f_name] = dic

        for r_name in robustness_names:
            r_dic = robustness_to_feature_to_dic[r_name]
            # Dump dictionary
            file_name = os.path.join(out_path, r_name + "_all.json")
            with open(file_name, 'w') as f:
                json.dump(r_dic, f, indent=2)

            # List best features
            items = list(r_dic.items())
            # sort by descending predictive power
            items = sorted(
                items,
                key=lambda x: r_dic[x[0]]["pred_score"],
                reverse=True
            )
            sim_s = ",".join(["_".join(s.split("_")[2:]) for s in similar_names])
            dissim_s = ",".join(["_".join(s.split("_")[2:]) for s in dissimilar_names])
            file_name = os.path.join(out_path, r_name + "_summary.csv")
            with open(file_name, 'w') as f:
                f.write("FeatureName,PredictivePower,{},{}\n".format(sim_s, dissim_s))
                for f_name, dic in items:
                    f.write("{},{}".format(f_name, dic["pred_score"]))
                    # similarity and dissimilarity scores
                    for i, s in enumerate(similar_names):
                        f.write(",{}".format(dic["sim_scores"][i]))
                    for i, s in enumerate(dissimilar_names):
                        f.write(",{}".format(dic["dissim_scores"][i]))
                    f.write("\n")

    def transform(self, X, y=None):
        """
        X and y are None. Data that is read is specified
        by 'self.streamer'.
        """
        smt_label = os.path.split(self.save_path)[-1]
        self.output_path = os.path.join(self.output_dir, smt_label)
        out_path = os.path.join(self.output_path, "robustness_measures")
        os.makedirs(out_path)

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

        # Compare robusntess of streamers with same diagnosis pair, but
        # same patient and different patient
        pairs = self.streamer_collection.get_diagnosis_pairs()
        for p in pairs:
            streamers = self.streamer_collection.diagnosis_pair_to_streamers(p)
            if len(streamers) < 2:
                continue

            assert len(streamers) == 2
            s1 = streamers[0]
            s2 = streamers[1]
            comp1 = streamer_to_comp[s1]
            comp2 = streamer_to_comp[s2]
            self.compare_computations(
                [comp1, comp2],
                [s1.name, s2.name],
                out_path
            )

        # Compare same patient pairs
        streamers = self.streamer_collection.get_same_patient_streamers()
        names = [s.name for s in streamers]
        comps = [streamer_to_comp[s] for s in streamers]
        self.compare_computations(comps, names, out_path)
        # Compare different patient pairs
        streamers = self.streamer_collection.get_different_patient_streamers()
        names = [s.name for s in streamers]
        comps = [streamer_to_comp[s] for s in streamers]
        self.compare_computations(comps, names, out_path)

        # Compute predictive power of features
        self.robustness_ratio(streamer_to_comp)

        # Make pickable
        self.streamers = None
        self.streamer_collection = None
