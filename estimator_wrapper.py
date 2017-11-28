import tensorflow as tf
import pandas as pd
from workflow.parsing import get_example_structure, MyParser
from workflow.input import Input, get_input_fns
from workflow.model_functions import ModelFunction
import time
import os
import sys
import numpy as np
import subprocess
from ml_project.configparse import ConfigParser
import argparse
from abc import ABC, abstractmethod
from pprint import pprint
import shutil
from sklearn.externals import joblib
import hashlib

#tf.logging.set_verbosity(tf.logging.DEBUG)

class Action(ABC):
  """docstring for Action"""
  def __init__(self, args):
    super(Action, self).__init__()
    self.args = args
    self.make_save_folder()

  def make_save_folder(self):
      if self.args.smt_label != "debug":
        basename = self.args.smt_label
      else:
        basename = time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-debug"

      path = os.path.join("data", basename)
      os.mkdir(os.path.normpath(path))
      self.save_path = path

  @abstractmethod
  def save(self):
    pass

class ConfigAction(Action):
  """docstring for ConfigAction"""
  def __init__(self, args, config):
    super(ConfigAction, self).__init__(args)
    self.config = ConfigParser().parse(config)
    self.pprint_config()

    getattr(self, self.args.action)()
    
    self.save()

  def fit(self):
    print("SHA1 checksums for inputs:")
    for file in self.config["data"]["train"]["files"]:
      subprocess.run(["sha1sum", file])

    model_fn = self.config["model_fn"](outdir=self.save_path, hooks=self.config["hooks"]).model_fn
    config = tf.estimator.RunConfig(model_dir=self.save_path, **self.config["config"])
    self.estimator = tf.estimator.Estimator(model_fn=model_fn, params=self.config["params"], config=config)

    input_fns = get_input_fns(self.config["data"])

    self.estimator.train(input_fn=input_fns["train"])

  def save(self):
    example_structure = get_example_structure(self.config["data"]["train"]["files"][0])
    feature_spec = {}
    for key, val in example_structure.items():
      if "input" in key:
        shape = example_structure[key]["shape"]
        dtype = getattr(tf, example_structure[key]["dtype"])
        feature_spec[key] = tf.placeholder(dtype=dtype, shape=[None] + shape)

    serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)

    model_save_path = self.estimator.export_savedmodel(self.save_path, serving_input_receiver_fn)

    print("Model saved to {}".format(model_save_path))

  def pprint_config(self):
      print("\n=========== Config ===========")
      pprint(self.config)
      print("==============================\n")
      sys.stdout.flush()


class ModelAction(Action):
  """docstring for ModelAction"""
  def __init__(self, args):
    super(ModelAction, self).__init__(args)
    getattr(self, self.args.action)()


  def predict(self):
    predictor = tf.contrib.predictor.from_saved_model(self.args.model)

    data = joblib.load(self.args.data)

    predictions = predictor(data)

    print("Saving predictions.csv to {}".format(self.save_path))

    df = pd.DataFrame(predictions)
    df.index += 1
    df.index.name = "ID"
    df.to_csv(os.path.join(self.save_path, "predictions.csv"))
  
  def save(self):
    pass


if __name__ == '__main__':
  arg_parser = argparse.ArgumentParser(description="Tensorflow runner.")
  arg_parser.add_argument("-C", "--config", help="config file")
  arg_parser.add_argument("-M", "--model", help="model file")

  arg_parser.add_argument("-D", "--data", help="Input data")

  arg_parser.add_argument("-a", "--action", choices=["transform", "predict",
                          "fit", "fit_transform", "score", "predict_proba"],
                          help="Action to perform.",
                          required=True)

  arg_parser.add_argument("smt_label", nargs="?", default="debug")

  args = arg_parser.parse_args()

  if args.config is None:
      ModelAction(args)
  else:
      ConfigAction(args, args.config)
