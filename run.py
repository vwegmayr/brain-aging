"""Scikit runner"""
import numpy as np
import argparse
import os
import sys
import pandas as pd
import csv

from sklearn.externals import joblib
from abc import ABC
from abc import abstractmethod
from ml_project import configparse
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
        if self.args.X is not None and self.args.y is not None:
            self.X, self.y = self._load_data()
        self.save_path = self._mk_save_folder()
        self.X_new, self.y_new = None, None
        self._X_new_set, self._y_new_set = False, False

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
        if self.args.smt_label != "debug":
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
        else:
            y = None
        return X, y

    def _mk_save_folder(self):
        if self.args.smt_label != "debug":
            basename = self.args.smt_label
            path = "data/"+basename+"/"
            os.mkdir(normpath(path))
            return path
        else:
            return None

    def transform(self):
        if "y" in getfullargspec(self.model.transform).args:
            self.X_new = self.model.transform(self.X, self.y)
        else:
            self.X_new = self.model.transform(self.X)
        self._X_new_set = True


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
        if "data" in self.config and self.args.X is None and self.args.y is None:
            self.model.fit(self.config["data"]["class"]())
        elif self.args.X is not None and self.args.y is None:
            self.model.fit(self.X, self.y)

    def fit_transform(self):
        self.fit()
        self.transform()

    def _save(self):
        class_name = self.config["class"].__name__
        joblib.dump(self.model, normpath(self.save_path+class_name+".pkl"))

        if self._X_new_set:
            path = self.save_path+"X_new.npy"
            np.save(normpath(path), self.X_new)

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
        self._y_new_set = True

    def predict_proba(self):
        self.y_new = self.model.predict_proba(self.X)
        self._y_new_set = True

    def score(self):
        self.model.score(self.X, self.y)

    def _save(self):
        y_path = normpath(self.save_path+"y_"+self.args.smt_label+".csv")
        X_path = normpath(self.save_path+"X_new.npy")
        if self._X_new_set:
            np.save(X_path, self.X_new)
        if self._y_new_set and self.args.action == "predict":
            df = pd.DataFrame({"Prediction": self.y_new})
            df.index += 1
            df.index.name = "ID"
            df.to_csv(y_path)
        elif self._y_new_set and self.args.action == "predict_proba":
            with open(y_path, "w") as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow(["ID", "Prediction"])
                n = 1
                for prediction in self.y_new:
                    prediction = np.round(prediction, decimals=4)
                    prediction = " ".join(prediction.astype("str"))
                    writer.writerow([n, prediction])
                    n += 1

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
        config_parser = configparse.ConfigParser()
        config = config_parser.parse_config(args.config)
        ConfigAction(args, config)
