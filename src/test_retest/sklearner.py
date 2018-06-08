from sklearn.metrics import accuracy_score
import importlib
import abc
from itertools import product
import pandas as pd
import numpy as np


def str_to_func(s):
    mod_name = ".".join(s.split(".")[:-1])
    mod = importlib.import_module(mod_name)
    f = getattr(mod, s.split(".")[-1])
    return f


def create_object(config):
    _class = config["class"]
    _params = config["params"]
    return _class(**_params)


class Sklearner(object):
    def __init__(self, config, estimator_config, streamer_config):
        self.config = config
        self.est = create_object(estimator_config)
        self.streamer = create_object(streamer_config)

    def fit(self, X, y):
        # Read true data based on streamer
        X, y = self.streamer.get_data_matrices(train=True)
        self.est.fit(X, y)

        # Compute score on test set
        X_test, y_test = self.streamer.get_data_matrices(train=False)
        pred = self.est.predict(X_test)

        sc_funcs = self.get_score_funcs()
        for sc_func in sc_funcs:
            sc = sc_func(y_test, pred)
            print("{}: {}".format(sc_func.__name__, sc))

        self.streamer = None

    def get_score_funcs(self):
        func_names = self.config["score_funcs"]
        return [str_to_func(s) for s in func_names]


class SklearnEvaluate(object):
    """
    Fit multiple sklearns with different parameters and compute
    scores.
    """
    def __init__(self, estimators, score_funcs):
        self.estimators = estimators
        self.score_funcs = [
            str_to_func(s) for s in score_funcs
        ]

    def set_save_path(self, path):
        self.save_path = path

    @abc.abstractmethod
    def get_train_data(self):
        pass

    @abc.abstractmethod
    def get_test_data(self):
        pass

    @abc.abstractmethod
    def tear_down(self):
        pass

    def fit(self, X=None, y=None):
        X_train, y_train = self.get_train_data()
        X_test, y_test = self.get_test_data()

        # Record all scores
        score_names = [f.__name__.split(".")[-1] for f in self.score_funcs]
        score_names = [s.split("_")[0] for s in score_names]
        self.rows = []

        for est_conf in self.estimators:
            # create estimator
            _class = est_conf["class"]
            _params = est_conf["params"]
            search = est_conf["search"]
            param_lists = []
            # build products of all possible parameters
            for param, values in search.items():
                # Replace None values
                for i, v in enumerate(values):
                    if v == 'None':
                        values[i] = None
                param_pairs = list(product([param], values))
                param_lists.append(param_pairs)

            combos = product(*param_lists)
            # score estimator for different parameters
            for combo in combos:
                for param, val in combo:
                    _params[param] = val

                est = _class(**_params)
                est.fit(X_train, y_train)
                train_acc = np.mean(y_train == est.predict(X_train))
                print("Train acc: {}".format(train_acc))
                self.score_estimator(est, X_test, y_test, str(combo))

        # build dataframe with score and dump it as csv
        self.df = pd.DataFrame(
            data=np.array(self.rows),
            columns=["Est", "para"] + score_names
        )
        self.df.to_csv(self.save_path + "/" + "scores.csv", index=False)
        self.df.to_latex(self.save_path + "/" + "scores.tex", index=False)
        self.tear_down()

    def score_estimator(self, est, X_test, y_test, params_s):
        pred = est.predict(X_test)
        est_name = est.__class__.__name__
        scores = []
        # collect scores
        for func in self.score_funcs:
            sc = func(y_test, pred)
            scores.append(round(sc, 4))

        row = [est_name, params_s] + scores
        self.rows.append(row)


class SklearnStreamerEvaluate(SklearnEvaluate):
    """
    Uses a streamer to get train and test data.
    """
    def __init__(self, streamer_config, *args, **kwargs):
        super(SklearnStreamerEvaluate, self).__init__(
            *args,
            **kwargs
        )
        self.streamer = create_object(streamer_config)

    def get_train_data(self):
        return self.streamer.get_data_matrices(train=True)

    def get_test_data(self):
        return self.streamer.get_data_matrices(train=False)

    def tear_down(self):
        self.streamer = None
