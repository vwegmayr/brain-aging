import yaml
import os
import copy
import pydoc
import tensorflow as tf
import numpy as np
from sklearn.metrics import recall_score, precision_score, \
    accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

from src.test_retest.mri.supervised_features import SliceClassification
from src.data.streaming.base import Group
from src.logging import MetricLogger


def predict_probabilities(est, input_fn):
    preds = est.predict(input_fn, ["probs"])
    res = []
    for pred in preds:
        res.append(pred["probs"][1])  # probability of being AD

    return np.array(res)


def specificity_score_deprecated(y_true, y_pred):
    """
    Compute true negative rate.
    TN / (TN + FP)
    """
    TN = 0
    FP = 0
    for y_t, y_p in zip(y_true, y_pred):
        if y_t == 0 and y_p == 0:
            TN += 1
        if y_t == 0 and y_p == 1:
            FP += 1

    if TN + FP == 0:
        return 0
    return TN / (TN + FP)


def specificity_score(y_true, y_pred):
    return recall_score(1 - y_true, 1 - y_pred)


def compute_scores(y_true, y_pred):
    funcs = [accuracy_score, recall_score, precision_score,
             specificity_score, f1_score]
    scores = {}
    names = []
    for f in funcs:
        s = f(y_true, y_pred)
        fname = f.__name__.split("_")[0]
        scores[fname] = round(s, 5)
        names.append(fname)

    return names, scores


def threshold_probs(labels, probs, target_metric, all_eps):
    target_scores = []
    all_scores = []
    for eps in all_eps:
        preds = (probs > eps).astype(np.float32)

        score_names, scores = compute_scores(labels, preds)
        target_scores.append(scores[target_metric])
        all_scores.append(scores)

    i = np.argmax(target_scores)
    return all_eps[i], all_scores[i]


def threshold_diff(labels, t0_probs, vagan_probs, target_metric, eps=None):
    if eps is None:
        all_eps = np.linspace(-1, 1, 200)
    else:
        all_eps = [eps]

    expected = np.array(labels)
    diffs = vagan_probs - t0_probs

    return threshold_probs(
        labels=expected,
        probs=diffs,
        target_metric=target_metric,
        all_eps=all_eps
    )


def threshold_log_ratio(labels, t0_probs, vagan_probs, target_metric, eps=None):

    expected = np.array(labels)
    ratios = np.log(vagan_probs / t0_probs)

    if eps is None:
        all_eps = ratios
    else:
        all_eps = [eps]

    return threshold_probs(
        labels=expected,
        probs=ratios,
        target_metric=target_metric,
        all_eps=all_eps
    )


def threshold_time_probs(labels, t1_probs, target_metric, eps=None):
    if eps is None:
        all_eps = np.linspace(-1, 1, 200)
    else:
        all_eps = [eps]

    expected = np.array(labels)

    return threshold_probs(
        labels=expected,
        probs=t1_probs,
        target_metric=target_metric,
        all_eps=all_eps
    )


class TwoStepConversion(object):
    def __init__(self, vagan_label, clf_label, split_paths, conversion_delta,
                 vagan_rescale, target_metric):
        """
        Args:
            - vagan_label: sumatra label for VAGAN record
            - clf_label: sumatra label for classifier label
            - split_paths: list of paths to split folders containg
              train-val-test split
        """
        self.vagan_label = vagan_label
        self.clf_label = clf_label
        self.split_paths = split_paths
        self.conversion_delta = conversion_delta
        self.vagan_rescale = vagan_rescale
        self.target_metric = target_metric

        self.load_models()

    def set_save_path(self, save_path):
        self.save_path = save_path

    def get_config(self):
        return {
            "vagan_label": self.vagan_label,
            "clf_label": self.clf_label,
            "split_path": self.split_paths,
            "conversion_delta": self.conversion_delta,
            "vagan_rescale": self.vagan_rescale,
            "target_metric": self.target_metric
        }

    def load_models(self):
        # Load config file
        self.conversion_key = "mci_ad_conv_delta_{}".format(
            self.conversion_delta
        )

        self.clf_folder = os.path.join("data", self.clf_label)
        with open(os.path.join(self.clf_folder, "config.yaml"), "r") as f:
            config = yaml.load(f)

        self.vagan_steps = self.conversion_delta
        self.load_classifier(config)
        self.load_vagan(config)

    def load_classifier(self, clf_config):
        clf_only_config = copy.deepcopy(clf_config)
        clf_only_config["params"]["streamer"]["class"] = \
            "src.data.streaming.mri_streaming.MRIConversionSingleStream"
        clf_only_config["params"]["streamer"]["class"] = \
            pydoc.locate(clf_only_config["params"]["streamer"]["class"])
        clf_only_config["params"]["streamer"]["params"]["stream_config"]["conversion_delta"] = \
            self.conversion_delta
        clf_only_config["params"]["streamer"]["params"]["stream_config"]["conversion_key"] = \
            self.conversion_key
        clf_only_config["params"]["streamer"]["params"]["stream_config"]["use_diagnoses"] = \
            ["health_mci", "health_ad"]

        self.clf_only_obj = SliceClassification(**clf_only_config["params"])
        self.clf_only_est = tf.estimator.Estimator(
            model_fn=self.clf_only_obj.model_fn,
            model_dir=self.clf_folder,
            params=clf_only_config["params"]["params"]
        )

    def load_vagan(self, clf_config):
        clf_vagan_config = copy.deepcopy(clf_config)
        clf_vagan_config["params"]["streamer"]["class"] = \
            "src.data.streaming.vagan_preprocessing.VaganConversionFarPredictions"
        clf_vagan_config["params"]["streamer"]["class"] = \
            pydoc.locate(clf_vagan_config["params"]["streamer"]["class"])
        clf_vagan_config["params"]["streamer"]["params"]["stream_config"]["conversion_delta"] = \
            self.conversion_delta
        clf_vagan_config["params"]["streamer"]["params"]["stream_config"]["conversion_key"] = \
            "mci_ad_conv_delta_4"
        clf_vagan_config["params"]["streamer"]["params"]["stream_config"]["vagan_steps"] = \
            self.vagan_steps
        clf_vagan_config["params"]["streamer"]["params"]["stream_config"]["vagan_label"] = \
            self.vagan_label
        clf_vagan_config["params"]["streamer"]["params"]["stream_config"]["cache_preprocessing"] = \
            False
        clf_vagan_config["params"]["streamer"]["params"]["stream_config"]["use_diagnoses"] = \
            ["health_mci", "health_ad"]
        clf_vagan_config["params"]["streamer"]["params"]["stream_config"]["vagan_rescale"] = \
            self.vagan_rescale

        self.clf_vagan_obj = SliceClassification(**clf_vagan_config["params"])
        self.clf_vagan_est = tf.estimator.Estimator(
            model_fn=self.clf_vagan_obj.model_fn,
            model_dir=self.clf_folder,
            params=clf_vagan_config["params"]["params"]
        )

    def fit(self, X, y=None):
        all_scores = {}
        logger = MetricLogger(self.save_path)

        for split_path in self.split_paths:
            scores = self.fit_split(split_path)

            for strat in scores.keys():
                if strat not in all_scores:
                    all_scores[strat] = {}
                    for k in scores[strat].keys():
                        all_scores[strat][k] = []

                for k, v in scores[strat].items():
                    all_scores[strat][k].append(v)

        print("++++++++++++++++++++")
        print(self.get_config())
        for strat, agg in all_scores.items():
            print(strat)
            for k, values in agg.items():
                print("{}: mean={}, std={}, median={}".format(
                    k,
                    np.mean(values),
                    np.std(values),
                    np.median(values)
                ))
                kk = strat + "_" + k
                logger.add_evaluations(
                    namespace=None,
                    evaluation_dic={
                        kk + "_mean": np.mean(values),
                        kk + "_std": np.std(values),
                        kk + "_var": np.var(values),
                        kk + "_median": np.median(values),
                    }
                )
        print("++++++++++++++++++++")
        logger.dump()

    def load_split(self, split_path):
        folder = split_path

        def load(fname):
            with open(os.path.join(folder, fname), 'r') as f:
                fids = [line.strip() for line in f]

            return fids

        train_ids = load("train.txt")
        val_ids = load("validation.txt")
        test_ids = load("test.txt")

        return train_ids, val_ids, test_ids

    def get_patient_id(self, fid):
        return self.clf_only_obj.streamer.get_patient_id(fid)

    def get_exact_age(self, fid):
        return self.clf_only_obj.streamer.get_exact_age(fid)

    def check_t0_t1_ordering(self, t0_ids, t1_ids):
        for t0, t1 in zip(t0_ids, t1_ids):
            p0 = self.get_patient_id(t0)
            p1 = self.get_patient_id(t1)
            assert p0 == p1

            a0 = self.get_exact_age(t0)
            a1 = self.get_exact_age(t1)
            assert a1 - a0 >= self.conversion_delta

    def get_labels(self, t0_fids):
        return np.array(
            [self.clf_only_obj.streamer.get_meta_info_by_key(fid, self.conversion_key)
             for fid in t0_fids]
        )

    def compute_probs(self, t0_ids, t1_ids):
        t0_batches = [Group([fid]) for fid in t0_ids]
        t1_batches = [Group([fid]) for fid in t1_ids]

        t0_input_fn = self.clf_only_obj.streamer.get_input_fn_for_groups(
            t0_batches
        )
        t1_input_fn = self.clf_only_obj.streamer.get_input_fn_for_groups(
            t1_batches
        )
        vagan_input_fn = self.clf_vagan_obj.streamer.get_input_fn_for_groups(
            t0_batches
        )

        return [
            predict_probabilities(self.clf_only_est, t0_input_fn),
            predict_probabilities(self.clf_only_est, t1_input_fn),
            predict_probabilities(self.clf_vagan_est, vagan_input_fn)
        ]

    def print_label_stats(self, labels):
        print("number samples: {}".format(len(labels)))
        print("class 0: {}".format(np.mean((labels == 0).astype(np.int32))))

    def add_scores(self, src, dest, namespace):
        for k, v in src.items():
            dest[namespace + "_" + k] = v

    def fit_split(self, split_path):
        train_ids, val_ids, test_ids = self.load_split(split_path)
        train_ids = train_ids + val_ids

        # Get t0 and t1 file IDs
        t0_train_ids = self.clf_only_obj.streamer.select_file_ids(train_ids)
        t1_train_ids = self.clf_only_obj.streamer.t1_fids[:]

        t0_test_ids = self.clf_only_obj.streamer.select_file_ids(test_ids)
        t1_test_ids = self.clf_only_obj.streamer.t1_fids[:]

        # Check pair ordering
        self.check_t0_t1_ordering(t0_train_ids, t1_train_ids)
        self.check_t0_t1_ordering(t0_test_ids, t1_test_ids)

        # Get labels
        train_labels = self.get_labels(t0_train_ids)
        test_labels = self.get_labels(t0_test_ids)

        print("++++++ Train stats:")
        self.print_label_stats(train_labels)

        print("++++++ Test stats:")
        self.print_label_stats(test_labels)

        # Compute probabilities
        t0_train_probs, t1_train_probs, vagan_train_probs = self.compute_probs(
            t0_train_ids, t1_train_ids
        )
        t0_test_probs, t1_test_probs, vagan_test_probs = self.compute_probs(
            t0_test_ids, t1_test_ids
        )

        # Compute scores
        # Threshold diff
        scores = {}
        best_eps, train_scores = threshold_diff(
            train_labels, t0_train_probs, vagan_train_probs, self.target_metric
        )
        _, test_scores = threshold_diff(
            test_labels, t0_test_probs, vagan_test_probs, self.target_metric, eps=best_eps
        )

        scores["thresh_diff"] = {
            "best_train_eps": best_eps,
        }
        self.add_scores(
            dest=scores["thresh_diff"],
            src=train_scores,
            namespace="train"
        )
        self.add_scores(
            dest=scores["thresh_diff"],
            src=test_scores,
            namespace="test"
        )

        # Threshold log ratio
        scores = {}
        best_eps, train_scores = threshold_log_ratio(
            train_labels, t0_train_probs, vagan_train_probs, self.target_metric
        )
        _, test_scores = threshold_log_ratio(
            test_labels, t0_test_probs, vagan_test_probs, self.target_metric, eps=best_eps
        )

        scores["thresh_log_ratio"] = {
            "best_train_eps": best_eps,
        }
        self.add_scores(
            dest=scores["thresh_log_ratio"],
            src=train_scores,
            namespace="train"
        )
        self.add_scores(
            dest=scores["thresh_log_ratio"],
            src=test_scores,
            namespace="test"
        )

        # Threshold t1
        best_eps, train_scores = threshold_time_probs(
            train_labels, vagan_train_probs, self.target_metric
        )

        _, test_scores = threshold_time_probs(
            test_labels, vagan_test_probs, self.target_metric, eps=best_eps
        )

        scores["thresh_t1"] = {
            "best_train_eps": best_eps,
        }

        self.add_scores(
            dest=scores["thresh_t1"],
            src=train_scores,
            namespace="train"
        )
        self.add_scores(
            dest=scores["thresh_t1"],
            src=test_scores,
            namespace="test"
        )

        # Treshold t0
        best_eps, train_scores = threshold_time_probs(
            train_labels, t0_train_probs, self.target_metric
        )

        _, test_scores = threshold_time_probs(
            test_labels, t0_test_probs, self.target_metric, eps=best_eps
        )

        scores["thresh_t0"] = {
            "best_train_eps": best_eps,
        }

        self.add_scores(
            dest=scores["thresh_t0"],
            src=train_scores,
            namespace="train"
        )
        self.add_scores(
            dest=scores["thresh_t0"],
            src=test_scores,
            namespace="test"
        )

        # GT
        best_eps, test_scores = threshold_diff(
            test_labels, t0_test_probs, t1_test_probs, self.target_metric
        )

        scores["thresh_diff_gt"] = {
            "best_test_eps": best_eps
        }
        self.add_scores(
            dest=scores["thresh_diff_gt"],
            src=test_scores,
            namespace="test"
        )

        best_eps, test_scores = threshold_log_ratio(
            test_labels, t0_test_probs, t1_test_probs, self.target_metric
        )

        scores["thresh_log_ratio_gt"] = {
            "best_test_eps": best_eps
        }
        self.add_scores(
            dest=scores["thresh_log_ratio_gt"],
            src=test_scores,
            namespace="test"
        )

        best_eps, test_scores = threshold_time_probs(
            test_labels, t1_test_probs, self.target_metric
        )

        scores["thresh_t1_gt"] = {
            "best_test_eps": best_eps
        }

        self.add_scores(
            dest=scores["thresh_t1_gt"],
            src=test_scores,
            namespace="test"
        )

        return scores


def ProbabilityConvergence(TwoStepConversion):
    def __init__(self, vagan_label, clf_label, split_path, time_delta,
                 conversion_delta, vagan_rescale):
        self.vagan_label = vagan_label
        self.clf_label = clf_label
        self.split_path = split_path
        self.time_delta = time_delta
        self.conversion_delta = conversion_delta
        self.vagan_rescale = vagan_rescale

        self.load_models()

    def set_save_path(self, save_path):
        self.save_path = save_path

    def compute_probs(self, t0_ids):
        n = len(t0_ids)
        probs = np.zeros(n, self.time_delta + 1)
        t0_batches = [Group([fid]) for fid in t0_ids]

        for i in range(self.time_delta):
            vagan_input_fn = self.clf_vagan_obj.streamer.get_input_fn_for_groups(
                t0_batches,
                vagan_steps=i + 1
            )
            probs_i = predict_probabilities(self.clf_vagan_est, vagan_input_fn)
            probs[:, i + 1] = probs_i

        # t0 probs
        t0_input_fn = self.clf_only_obj.streamer.get_input_fn_for_groups(
            t0_batches
        )
        t0_probs = predict_probabilities(self.clf_only_est, t0_input_fn)
        probs[:, 0] = t0_probs

        return probs

    def get_hc_ids(self, fids):
        return [fid for fid in fids
                if self.clf_only_obj.streamer.get_diagnose(fid) == "healthy"]

    def get_conv_ids(self, fids):
        k = "mci_ad_conv_delta_{}".format(self.conversion_delta)
        return [fid for fid in fids
                if self.clf_only_obj.streamer.get_meta_info_by_key(fid, k) == 1]

    def get_non_conv_ids(self, fids):
        k = "mci_ad_conv_delta_{}".format(self.conversion_delta)
        return [fid for fid in fids
                if self.clf_only_obj.streamer.get_meta_info_by_key(fid, k) == 0]

    def fit(self, X, y=None):
        _, val_ids, test_ids = self.load_split(self.split_path)
        t0_val_ids = self.clf_only_obj.streamer.select_file_ids(val_ids)
        t0_test_ids = self.clf_only_obj.streamer.select_file_ids(test_ids)

        hc_ids = self.get_hc_ids(t0_val_ids)
        conv_ids = self.get_conv_ids(t0_test_ids)
        non_conv_ids = self.get_non_conv_ids(t0_test_ids)

        hc_probs = self.compute_probs(hc_ids)
        conv_probs = self.compute_probs(conv_ids)
        non_conv_probs = self.compute_probs(non_conv_ids)

        x = list(range(self.time_delta + 1))
        plt.plot(x, hc_probs, label="HC")
        plt.plot(x, conv_probs, label="Converting")
        plt.plot(x, non_conv_probs, label="Non-Converting")

        plt.legend(loc=0, ncol=1)
        plt.show()
        plt.savefig(os.path.join(self.save_path, "population.pdf"))
