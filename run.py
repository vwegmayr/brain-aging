"""Scikit runner"""
import numpy as np
import argparse
import os
import sys
import pandas as pd
import csv
import time

from sklearn.externals import joblib
from abc import ABC, abstractmethod
from ml_project.configparse import ConfigParser
from pprint import pprint
from os.path import normpath
from inspect import getfullargspec


class Action(ABC):
    """Abstract Action class

    Args:
        args (Namespace): Parsed arguments
    """
    def __init__(self, args):
        self.args = args
        self._check_action(args.action)
        if self.args.X is not None:
            self.X, self.y = self._load_data()
        self._mk_save_folder()
        self.X_new, self.y_new = None, None

    @abstractmethod
    def _save(self):
        pass

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def _check_action(self):
        pass

    def act(self):
        self.model = self._load_model()
        getattr(self, self.args.action)()
        self._save()

    def _load_data(self):
        try:
            X = np.load(self.args.X)
        except FileNotFoundError:
            print("{} not found. "
                  "Please download data first.".format(self.args.X))
            exit()
        if self.args.y is not None:
            try:
                y = np.loadtxt(self.args.y)
            except FileNotFoundError:
                print("{} not found. "
                      "Please download data first.".format(self.args.y))
                exit()
            except UnicodeDecodeError:
                y = np.load(self.args.y)
        else:
            y = None
        return X, y

    def _mk_save_folder(self):
        if self.args.smt_label != "debug":
            self.time_stamp = self.args.smt_label
        else:
            self.time_stamp = (time.strftime(
                "%Y%m%d-%H%M%S",
                time.gmtime()) + "-debug")

        path = os.path.join("data", self.time_stamp)
        os.mkdir(os.path.normpath(path))

        self.save_path = path

    def transform(self):
        if "y" in getfullargspec(self.model.transform).args:
            self.X_new = self.model.transform(self.X, self.y)
        else:
            self.X_new = self.model.transform(self.X)


class ConfigAction(Action):
    """Class to handle config file actions

    Args:
        args (Namespace): Parsed arguments
        config (dict): Parsed config file

    """
    def __init__(self, args, config):
        super(ConfigAction, self).__init__(args)
        self.config = config
        self.pprint_config()
        self.act()

    def fit(self):
        if self.args.X is not None and self.args.y is not None:
            self.model.fit(self.X, self.y)
        else:
            print("Expected X and y for fit.")

    def fit_transform(self):
        self.fit()
        self.transform()

    def _save(self):
        class_name = self.config["class"].__name__
        model_path = normpath(os.path.join(self.save_path, class_name+".pkl"))
        joblib.dump(self.model, model_path)

        if self.X_new is not None:
            X_path = normpath(os.path.join(
                self.save_path,
                "X_" + self.time_stamp + ".npy"))
            np.save(X_path, self.X_new)

    def _load_model(self):
        if "params" in self.config:
            model = self.config["class"](**self.config["params"])
        else:
            model = self.config["class"]()

        if hasattr(model, "set_save_path"):
            model.set_save_path(self.save_path)

        return model

    def _check_action(self, action):
        if action not in ["fit", "fit_transform"]:
            raise RuntimeError("Can only run fit or fit_transform from config,"
                               " got {}.".format(action))

    def pprint_config(self):
        print("\n=========== Config ===========")
        pprint(self.config)
        print("==============================\n")
        sys.stdout.flush()


class ModelAction(Action):
    """Class to model actions

    Args:
        args (Namespace): Parsed arguments
    """
    def __init__(self, args):
        super(ModelAction, self).__init__(args)
        self.act()

    def predict(self):
        self.y_new = self.model.predict(self.X)

    def predict_proba(self):
        self.y_new = self.model.predict_proba(self.X)

    def score(self):
        self.model.score(self.X, self.y)

    def _save(self):
        y_path = normpath(os.path.join(
            self.save_path, "y_" + self.time_stamp + ".csv"))
        X_path = normpath(os.path.join(
            self.save_path, "X_" + self.time_stamp + ".npy"))

        if self.X_new is not None:
            np.save(X_path, self.X_new)

        if self.y_new is not None:
            if isinstance(self.y_new, (np.ndarray, list)):
                with open(y_path, "w") as csvfile:
                    writer = csv.writer(csvfile, delimiter=',')
                    writer.writerow(["ID", "Prediction"])
                    for id, prediction in enumerate(self.y_new):
                        prediction = np.round(prediction, decimals=6)
                        if len(prediction.shape) > 1:
                            prediction = " ".join(prediction.astype("str"))
                        writer.writerow([id+1, prediction])
            elif isinstance(self.y_new, dict):
                df = pd.DataFrame(self.y_new)
                df.index += 1
                df.index.name = "ID"
                df.to_csv(y_path)

    def _load_model(self):
        model = joblib.load(self.args.model)
        if hasattr(model, "set_save_path"):
            model.set_save_path(self.save_path)
        return model

    def _check_action(self, action):
        if action not in ["transform", "predict", "score", "predict_proba"]:
            raise RuntimeError("Can only run transform, predict, predict_proba"
                               " or score from model, got {}.".format(action))


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(description="Scikit runner.")

    arg_parser.add_argument("-C", "--config", help="config file")
    arg_parser.add_argument("-M", "--model", help="model file")

    arg_parser.add_argument("-X", help="Input data")
    arg_parser.add_argument("-y", help="Input labels")

    arg_parser.add_argument("-a", "--action", choices=["transform", "predict",
                            "fit", "fit_transform", "score", "predict_proba"],
                            help="Action to perform.",
                            required=True)

    arg_parser.add_argument("smt_label", nargs="?", default="debug")

    args = arg_parser.parse_args()

    if args.config is None:
        ModelAction(args)
    else:
        config = ConfigParser().parse(args.config)
        ConfigAction(args, config)
