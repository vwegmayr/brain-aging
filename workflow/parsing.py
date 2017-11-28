import tensorflow as tf
from google.protobuf.json_format import MessageToJson
import json
import warnings
import numpy as np
from abc import ABC, abstractmethod

def parser(record):
  keys_to_features = {
      "X": tf.FixedLenFeature(shape=[], dtype=tf.string),
      "y": tf.FixedLenFeature(shape=[], dtype=tf.int64),
  }
  parsed = tf.parse_single_example(record, features=keys_to_features)

  image = tf.decode_raw(parsed["X"], tf.float64)

  #image.set_shape([20])
  image = tf.reshape(image, [20])
  label = tf.cast(parsed["y"], tf.float64)
  #label.set_shape([])
  #label = tf.reshape(label, [-1])

  return {"X": image}, label


def get_example_structure(tf_record_file):
  iterator = tf.python_io.tf_record_iterator(path=tf_record_file)

  features = MessageToJson(tf.train.Example.FromString(next(iterator)))

  features = json.loads(features)["features"]["feature"]

  feature_structure = {}

  for feature_name, body in features.items():
    for list_type in ["int64List", "bytesList", "floatList"]:
      try:
        feature_value = body[list_type]
        feature_structure[feature_name] = {"type": list_type}
        if "shape" in feature_name:
          if list_type != "int64List":
            message = "Expected int64List for shape feature of {}, got {}"
            warnings.warn(message.format(feature_name.split("_")[0], list_type))
          feature_structure[feature_name]["value"] = [int(dim) for dim in feature_value["value"]]
        if "dtype" in feature_name:          
          feature_structure[feature_name]["value"] = "".join(map(lambda x: chr(int(x)), feature_value["value"]))
      except KeyError:
        pass

  for key, val in list(feature_structure.items()):
    var_key = key.split("_")[0]
    if "shape" in key:
      feature_structure[var_key]["shape"] = val["value"]
    if "dtype" in key:
      feature_structure[var_key]["dtype"] = val["value"]

  return feature_structure


def basic_parser(record, example_structure):
  keys_to_features = {}
  for key, val in example_structure.items():

    if val["type"] == "int64List":
      dtype = tf.int64
    elif val["type"] == "bytesList":
      dtype = tf.string
    elif va["type"] == "floatList":
      dytpe = tf.float32

    if "dtype" in key:
      keys_to_features[key] = tf.VarLenFeature(dtype=dtype)
    else:
      keys_to_features[key] = tf.FixedLenFeature(shape=[], dtype=dtype)

  parsed = tf.parse_single_example(record, features=keys_to_features)

  for key, val in example_structure.items():
    if val["type"] == "int64List":
      parsed[key] = tf.cast(parsed[key], tf.int64)
    elif val["type"] == "bytesList":
      parsed[key] = tf.decode_raw(parsed[key], getattr(tf, val["dtype"]))
    elif val["type"] == "floatList":
      parsed[key] = tf.cast(parsed[key], tf.float32)

  for key, val in example_structure.items():
    if "shape" in val:
      parsed[key] = tf.reshape(parsed[key], val["shape"])  

  return parsed


class Parser(ABC):
  """docstring for Parser"""
  def __init__(self, example_structure):
    super(Parser, self).__init__()
    self.example_structure = example_structure

  @abstractmethod
  def parse(self, record):
    pass
    

class MyParser(Parser):
  """docstring for MyParser"""
  def __init__(self, example_structure):
    super(MyParser, self).__init__(example_structure)

  def parse(self, record):
    basic_parsed = basic_parser(record, self.example_structure)

    return {"X": basic_parsed["X"]}, basic_parsed["y"]