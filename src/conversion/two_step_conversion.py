import yaml
import os
import copy
import pydoc
import tensorflow as tf
import numpy as np
from sklearn.metrics import recall_score, precision_score, \
    accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import itertools

from src.test_retest.mri.supervised_features import SliceClassification
from src.data.streaming.base import Group
from src.logging import MetricLogger
from src.baum_vagan.utils import ncc


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
        scores[fname] = round(s, 6)
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


def threshold_harmonic_all_probs(labels, all_probs, target_metric, eps=None, weights=None):
    if eps is None:
        all_eps = np.linspace(0, 1, 10)
    else:
        all_eps = [eps]

    best_score = -1
    best_all_scores = None
    best_eps = -1
    best_weights = []
    expected = np.array(labels)

    n_probs = all_probs.shape[1]
    weight_vals = np.linspace(0.01, 1, 10)

    mci_probs = 1 - all_probs

    if weights is None:
        all_combos = itertools.product(weight_vals, repeat=n_probs)
    else:
        all_combos = [weights]

    for combo in all_combos:
        # compute harmonic mean
        num = np.sum(mci_probs, axis=1)
        denom = np.sum(combo * (1 / mci_probs), axis=1)
        denom = denom + 0.000001
        harmonic_means = num / denom

        new_eps, all_scores = threshold_probs(
            labels=expected,
            probs=harmonic_means,
            target_metric=target_metric,
            all_eps=all_eps
        )
        score = all_scores[target_metric]

        if score > best_score:
            best_score = score
            best_all_scores = all_scores
            best_eps = new_eps
            best_weights = np.copy(combo)

    return best_eps, best_weights, best_all_scores


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


def threshold_max_t0_t1(labels, t0_probs, t1_probs, target_metric, eps=None):
    if eps is None:
        all_eps = np.linspace(-1, 1, 200)
    else:
        all_eps = [eps]

    expected = np.array(labels)
    all_probs = np.hstack((
        np.reshape(t0_probs, (-1, 1)),
        np.reshape(t1_probs, (-1, 1))
    ))
    maxis = np.max(all_probs, axis=1)

    return threshold_probs(
        labels=expected,
        probs=maxis,
        target_metric=target_metric,
        all_eps=all_eps
    )


def threshold_max_all(labels, all_probs, target_metric, eps=None):
    if eps is None:
        all_eps = np.linspace(-1, 1, 200)
    else:
        all_eps = [eps]

    expected = np.array(labels)
    maxis = np.max(all_probs, axis=1)

    return threshold_probs(
        labels=expected,
        probs=maxis,
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
                 vagan_rescale, target_metric, all_steps, n_iterations):
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
        self.all_steps = all_steps
        self.n_iterations = n_iterations

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
            header = []
            clf_name = self.clf_label.replace("/","_")
            header.append("method")
            header.append("clf_name")
            header.append("gan_name")
            header.append("metric")
            header.append("mean")
            header.append("std")
            header.append("median")
            for i in range(len(self.split_paths)):
                header.append("split_{}".format(i))
            rows = []
            for k, values in agg.items():
                row = [strat, clf_name, self.vagan_label]
                """
                print(values)
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
                """
                row.append(k)
                row.append(np.mean(values))
                row.append(np.std(values))
                row.append(np.median(values))
                for i, v in enumerate(values):
                    row.append(v)

                rows.append(row)

            df = pd.DataFrame(
                data=np.array(rows),
                columns=header
            )
            df = df.round(6)
            print(df.to_csv(index=False))
            df.to_csv(os.path.join(self.save_path, '{}_scores.csv'.format(strat)), index=False)
                
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
            t0_batches,
            vagan_steps=self.n_iterations
        )

        return [
            predict_probabilities(self.clf_only_est, t0_input_fn),
            predict_probabilities(self.clf_only_est, t1_input_fn),
            predict_probabilities(self.clf_vagan_est, vagan_input_fn)
        ]

    def compute_gt_probs(self, t0_ids, t1_ids):
        t0_batches = [Group([fid]) for fid in t0_ids]
        t1_batches = [Group([fid]) for fid in t1_ids]

        t0_input_fn = self.clf_only_obj.streamer.get_input_fn_for_groups(
            t0_batches
        )
        t1_input_fn = self.clf_only_obj.streamer.get_input_fn_for_groups(
            t1_batches
        )

        return [
            predict_probabilities(self.clf_only_est, t0_input_fn),
            predict_probabilities(self.clf_only_est, t1_input_fn),
        ]

    def compute_all_probs(self, t0_ids):
        n = len(t0_ids)
        probs = np.zeros((n, self.conversion_delta + 1))
        t0_batches = [Group([fid]) for fid in t0_ids]

        for i in range(self.conversion_delta):
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
        if self.all_steps:
            t0_train_probs, t1_train_probs = self.compute_gt_probs(
                t0_train_ids, t1_train_ids
            )
            t0_test_probs, t1_test_probs = self.compute_gt_probs(
                t0_test_ids, t1_test_ids
            )

            all_vagan_train_probs = self.compute_all_probs(t0_train_ids)
            all_vagan_test_probs = self.compute_all_probs(t0_test_ids)
            vagan_train_probs = all_vagan_train_probs[:, self.conversion_delta]
            vagan_test_probs = all_vagan_test_probs[:, self.conversion_delta]
        else:
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

        # Threshold max t0, t1
        best_eps, train_scores = threshold_max_t0_t1(
            train_labels, t0_train_probs, vagan_train_probs, self.target_metric
        )

        _, test_scores = threshold_max_t0_t1(
            test_labels, t0_test_probs, vagan_test_probs, self.target_metric, eps=best_eps
        )

        scores["thresh_max_t0_t1"] = {
            "best_train_eps": best_eps,
        }

        self.add_scores(
            dest=scores["thresh_max_t0_t1"],
            src=train_scores,
            namespace="train"
        )
        self.add_scores(
            dest=scores["thresh_max_t0_t1"],
            src=test_scores,
            namespace="test"
        )

        if self.all_steps:
            # Threshold max all
            best_eps, train_scores = threshold_max_all(
                train_labels, all_vagan_train_probs, self.target_metric
            )

            _, test_scores = threshold_max_all(
                test_labels, all_vagan_test_probs, self.target_metric, eps=best_eps
            )

            scores["thresh_max_all"] = {
                "best_train_eps": best_eps,
            }

            self.add_scores(
                dest=scores["thresh_max_all"],
                src=train_scores,
                namespace="train"
            )
            self.add_scores(
                dest=scores["thresh_max_all"],
                src=test_scores,
                namespace="test"
            )

            # Threshold harmonic
            best_eps, best_weights, train_scores = threshold_harmonic_all_probs(
                train_labels, all_vagan_train_probs, self.target_metric,
                eps=None, weights=None
            )
            print(">>>>>> harmonic weights")
            print(best_weights)

            _, _, test_scores = threshold_harmonic_all_probs(
                test_labels, all_vagan_test_probs, self.target_metric,
                eps=best_eps, weights=best_weights
            )

            scores["threshold_harmonic"] = {
                "best_train_eps": best_eps,
            }

            self.add_scores(
                dest=scores["thresh_harmonic"],
                src=train_scores,
                namespace="train"
            )
            self.add_scores(
                dest=scores["thresh_harmonic"],
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
        best_eps, train_scores = threshold_diff(
            train_labels, t0_train_probs, t1_train_probs, self.target_metric
        )

        _, test_scores = threshold_diff(
            test_labels, t0_test_probs, t1_test_probs, self.target_metric, eps=best_eps
        )

        scores["thresh_diff_gt"] = {
            "best_test_eps": best_eps
        }

        self.add_scores(
            dest=scores["thresh_diff_gt"],
            src=train_scores,
            namespace="train"
        )
        self.add_scores(
            dest=scores["thresh_diff_gt"],
            src=test_scores,
            namespace="test"
        )

        best_eps, train_scores = threshold_log_ratio(
            train_labels, t0_train_probs, t1_train_probs, self.target_metric
        )

        _, test_scores = threshold_log_ratio(
            test_labels, t0_test_probs, t1_test_probs, self.target_metric, eps=best_eps
        )

        scores["thresh_log_ratio_gt"] = {
            "best_test_eps": best_eps
        }
        self.add_scores(
            dest=scores["thresh_log_ratio_gt"],
            src=train_scores,
            namespace="train"
        )
        self.add_scores(
            dest=scores["thresh_log_ratio_gt"],
            src=test_scores,
            namespace="test"
        )

        best_eps, train_scores = threshold_time_probs(
            train_labels, t1_train_probs, self.target_metric
        )

        _, test_scores = threshold_time_probs(
            test_labels, t1_test_probs, self.target_metric, eps=best_eps
        )

        scores["thresh_t1_gt"] = {
            "best_test_eps": best_eps
        }

        self.add_scores(
            dest=scores["thresh_t1_gt"],
            src=train_scores,
            namespace="train"
        )

        self.add_scores(
            dest=scores["thresh_t1_gt"],
            src=test_scores,
            namespace="test"
        )

        best_eps, train_scores = threshold_time_probs(
            train_labels, t0_train_probs, self.target_metric
        )

        _, test_scores = threshold_time_probs(
            test_labels, t0_test_probs, self.target_metric, eps=best_eps
        )

        scores["thresh_t0_gt"] = {
            "best_test_eps": best_eps
        }

        self.add_scores(
            dest=scores["thresh_t0_gt"],
            src=train_scores,
            namespace="train"
        )

        self.add_scores(
            dest=scores["thresh_t0_gt"],
            src=test_scores,
            namespace="test"
        )

        best_eps, train_scores = threshold_max_t0_t1(
            train_labels, t0_train_probs, t1_train_probs, self.target_metric
        )

        _, test_scores = threshold_max_t0_t1(
            test_labels, t0_test_probs, t1_test_probs, self.target_metric, eps=best_eps
        )

        scores["thresh_max_t0_t1_gt"] = {
            "best_train_eps": best_eps,
        }

        self.add_scores(
            dest=scores["thresh_max_t0_t1_gt"],
            src=train_scores,
            namespace="train"
        )
        self.add_scores(
            dest=scores["thresh_max_t0_t1_gt"],
            src=test_scores,
            namespace="test"
        )

        return scores


class ProbabilityConvergence(TwoStepConversion):
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
        probs = np.zeros((n, self.time_delta + 1))
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
    
    def get_t0_ids(self, file_ids):
        streamer = self.clf_only_obj.streamer
        patient_groups = streamer.make_patient_groups(file_ids)
        res = []
        for g in patient_groups:
            fids = g.file_ids
            fids = sorted(fids, key=lambda x: streamer.get_exact_age(x))
            res.append(fids[0])
        
        return res

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
        t0_val_ids = self.get_t0_ids(val_ids)
        t0_test_ids = self.clf_only_obj.streamer.select_file_ids(test_ids)

        hc_ids = self.get_hc_ids(val_ids)
        conv_ids = self.get_conv_ids(t0_test_ids)
        non_conv_ids = self.get_non_conv_ids(t0_test_ids)

        hc_probs = self.compute_probs(hc_ids)
        conv_probs = self.compute_probs(conv_ids)
        non_conv_probs = self.compute_probs(non_conv_ids)
        
        np.save("hc_population.npy", hc_probs)
        np.save("converting_population.npy", conv_probs)
        np.save("non_converting_population.npy", non_conv_probs)

        x = list(range(self.time_delta + 1))
        hc_mean = np.mean(hc_probs, axis=0)
        hc_std = np.std(hc_probs, axis=0)
        conv_mean = np.mean(conv_probs, axis=0)
        conv_std = np.std(conv_probs, axis=0)
        non_conv_mean = np.mean(non_conv_probs, axis=0)
        non_conv_std = np.std(non_conv_probs, axis=0)
        plt.errorbar(x, hc_mean, yerr=hc_std, marker='o', label="HC")
        plt.errorbar(x, conv_mean, yerr=conv_std, marker='+', label="Converting")
        plt.errorbar(x, non_conv_mean, yerr=non_conv_std, marker='s', label="Non-Converting")

        plt.legend(loc=0, ncol=1)
        plt.show()
        #plt.savefig(os.path.join(self.save_path, "population.pdf"))

        # boxplot
        plt.boxplot(hc_probs)
        plt.title("HC box plot")
        plt.show()
        
        plt.boxplot(conv_probs)
        plt.title("Converting box plot")
        plt.show()
        
        plt.boxplot(non_conv_probs)
        plt.title("Non-Converting box plot")
        plt.show()


class NCCComputation(TwoStepConversion):
    def __init__(self, vagan_label, clf_label, split_path, conversion_delta, vagan_rescale):
        self.vagan_label = vagan_label
        self.clf_label = clf_label
        self.split_path = split_path
        self.conversion_delta = conversion_delta
        self.vagan_rescale = vagan_rescale

        self.load_models()

    def get_images(self, ids):
        streamer = self.clf_vagan_obj.streamer
        images = []
        for fid in ids:
            path = streamer.get_file_path(fid)
            im = streamer.load_sample(path).astype(np.float32)
            images.append(im)

        return images

    def get_fake_t1_images(self, t0_fids):
        vagan = self.clf_vagan_obj.streamer.wrapper.vagan
        t0_images = self.get_images(t0_fids)
        gen = []
        for t0_im in t0_images:
            images, masks = vagan.iterated_far_prediction(
                t0_im, self.conversion_delta
            )
            gen.append(images[-1])

        return gen

    def print_scores(self, scores):
        print("Mean {}".format(np.mean(scores)))
        print("STD {}".format(np.std(scores)))
        print("Median {}".format(np.median(scores)))
        print("5-percentile {}".format(np.percentile(scores, 5)))
        print("95-percentile {}".format(np.percentile(scores, 95)))
        
        return ["mean","std","median","5-perc", "95-perc", "min", "max"],\
            [np.mean(scores), np.std(scores), np.median(scores), np.percentile(scores, 5), np.percentile(scores, 95),
             np.min(scores), np.max(scores)]

    def fit(self, X, y=None):
        train_ids, val_ids, test_ids = self.load_split(self.split_path)
        all_ids = train_ids + val_ids + test_ids

        # Get t0 and t1 file IDs
        t0_ids = self.clf_only_obj.streamer.select_file_ids(all_ids)
        t1_ids = self.clf_only_obj.streamer.t1_fids[:]
        
        print(">>>>>>> {} images".format(len(t0_ids)))
        
        for t0, t1 in zip(t0_ids, t1_ids):
            s = self.clf_only_obj.streamer
            assert s.get_patient_id(t0) == s.get_patient_id(t1)
            assert s.get_exact_age(t1) - s.get_exact_age(t0) >= self.conversion_delta

        t0_images = self.get_images(t0_ids)
        fake_t1_images = self.get_fake_t1_images(t0_ids)
        real_t1_images = self.get_images(t1_ids)

        id_to_t0 = {}
        for fid, im in zip(t0_ids, t0_images):
            id_to_t0[fid] = im
        
        id_to_t1 = {}
        for fid, im in zip(t1_ids, real_t1_images):
            id_to_t1[fid] = im

        # NCC for difference maps
        scores = []
        for t0, fake, real in zip(t0_images, fake_t1_images, real_t1_images):
            gt_diff = real - t0
            gen_diff = fake - t0
            assert gt_diff.shape == gen_diff.shape
            scores.append(ncc(gt_diff, gen_diff))

        print("NCC scores t0/t1")
        names, values = self.print_scores(scores)
        
        header = ["pair_type"] + names
        rows = [["not_random"] + values]
        
        # random pairs
        pairs = list(itertools.product(t0_ids, t1_ids))
        self.clf_only_obj.streamer.np_random.shuffle(pairs)
        scores = []
        c = 0
        i = 0
        while c < len(t0_ids):
            s = self.clf_only_obj.streamer
            p0 = pairs[2 * i]
            p1 = pairs[2 * i + 1]
            
            i += 1
            if s.get_patient_id(p0[0]) == s.get_patient_id(p0[1]):
                continue
            if s.get_patient_id(p1[0]) == s.get_patient_id(p1[1]):
                continue
            
            c += 1
            diff0 = id_to_t1[p0[1]] - id_to_t0[p0[0]]
            diff1 = id_to_t1[p1[1]] - id_to_t0[p1[0]]
            scores.append(ncc(diff0, diff1))
            
        print("Random pairs")
        names, values = self.print_scores(scores)
        rows.append(['random'] + values)
        
        df = pd.DataFrame(
            data=np.array(rows),
            columns=header
        )
        print(df.to_csv(index=False))
