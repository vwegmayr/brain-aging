import tensorflow as tf
from workflow.parsing import get_example_structure, MyParser
from input import Input
from workflow.model_functions import ModelFunction
import time
import os
import numpy as np
import subprocess
from ml_project.configparse import ConfigParser


#tf.logging.set_verbosity(tf.logging.DEBUG)

folder = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
out_dir = os.path.join("data", folder)

model_fn = ModelFunction(out_dir=out_dir, hooks=[{"type": "StopAtStepHook", "params": {"num_steps": 37}}]).model_fn

estimator = tf.estimator.Estimator(model_fn=model_fn,
             params={"learning_rate": 0.05},
             config=tf.estimator.RunConfig(model_dir=out_dir, tf_random_seed=42))

example_structure = get_example_structure("data/tf.test.shape")

parser = MyParser(example_structure)

data_train = Input(tf_files=["data/tf.test.shape"],
                   parser=parser.parse)

print(example_structure)

estimator.train(input_fn=data_train.input_fn)

data_test = Input(tf_files=["data/tf.test.shape"],
                   parser=parser.parse)

for i, y in enumerate(estimator.predict(input_fn=data_test.input_fn)):
  if i < 5:
    print(y)

shape = example_structure["X"]["shape"]
feature_spec = {"X": tf.placeholder(dtype=tf.float64, shape=[None] + shape)}
serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)

save_path = estimator.export_savedmodel(out_dir, serving_input_receiver_fn)

print("Save Path: {}".format(save_path))

predict = tf.contrib.predictor.from_saved_model(save_path)

print(predict({"X": np.random.rand(3,20)}))

subprocess.run(["saved_model_cli", "run", "--dir", save_path, "--tag_set", "serve", 
  "--signature_def", "ages", "--input_exprs", "X=np.random.rand(3,20)"])


if __name__ == '__main__':
  arg_parser = argparse.ArgumentParser(description="Tensorflow runner.")
  arg_parser.add_argument("-C", "--config", help="config file")

  config_parser = configparse.ConfigParser()
  config = config_parser.parse_config(args.config)