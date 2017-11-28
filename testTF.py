import tensorflow as tf
import time
import os
from tensorflow.python.framework import ops

#tf.logging.set_verbosity(tf.logging.DEBUG)

def model_fn(features, labels, mode, params, config):
  """Model function for Estimator."""

  # Connect the first hidden layer to input layer
  # (features["x"]) with relu activation
  first_hidden_layer = tf.layers.dense(features["X"], 10, activation=tf.nn.relu, name="layer0")

  tf.summary.image("first_hidden_layer", tf.reshape(first_hidden_layer,[-1,10,1,1]))

  # Connect the second hidden layer to first hidden layer with relu
  second_hidden_layer = tf.layers.dense(
      first_hidden_layer, 10, activation=tf.nn.relu)

  # Connect the output layer to second hidden layer (no activation fn)
  output_layer = tf.layers.dense(second_hidden_layer, 1)

  # Reshape output layer to 1-dim Tensor to return predictions
  predictions = tf.reshape(output_layer, [-1])

  # Provide an estimator spec for `ModeKeys.PREDICT`.
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"ages": predictions},
        export_outputs={"ages": tf.estimator.export.PredictOutput({"ages": predictions})})

  # Calculate loss using mean squared error
  loss = tf.losses.mean_squared_error(labels, predictions)

  #loss_op = tf.summary.scalar("loss", loss, collections=ops.GraphKeys.SUMMARIES)
  tf.summary.scalar("loss", loss)

  # Calculate root mean squared error as additional eval metric
  eval_metric_ops = {
      "rmse": tf.metrics.root_mean_squared_error(
          tf.cast(labels, tf.float64), predictions)
  }

  optimizer = tf.train.GradientDescentOptimizer(
      learning_rate=params["learning_rate"])
  train_op = optimizer.minimize(
      loss=loss, global_step=tf.train.get_global_step())

  hooks=[tf.train.LoggingTensorHook({"first_layer": tf.reduce_sum(first_hidden_layer)}, every_n_iter=50),
         tf.train.SummarySaverHook(summary_op=tf.summary.scalar("first_layer", tf.reduce_sum(first_hidden_layer)), output_dir=params["out_dir"], save_steps=10)]

  print(locals())

  # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops,
      training_hooks=hooks)


record_iterator = tf.python_io.tf_record_iterator(path="data/tf.test")

string_record = next(record_iterator)
example = tf.train.Example()
example.ParseFromString(string_record)
shape = list(example.features.feature['shape'].int64_list.value)

print(shape)

def parser(record):
  keys_to_features = {
      "X": tf.FixedLenFeature(shape=[20], dtype=tf.string),
      "y": tf.FixedLenFeature(shape=[], dtype=tf.int64),
  }
  parsed = tf.parse_single_example(record, features=keys_to_features)

  image = tf.decode_raw(parsed["X"], tf.float64)

  #image.set_shape([20])
  #image = tf.reshape(image, shape)
  label = tf.cast(parsed["y"], tf.float64)
  #label.set_shape([])
  #label = tf.reshape(label, [-1])

  return {"X": image}, label

def train_input_fn():
  dataset = tf.data.TFRecordDataset(["data/tf.test"])
  dataset = dataset.map(parser)
  dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.batch(10)
  dataset = dataset.repeat(10)
  iterator = dataset.make_one_shot_iterator()

  return iterator.get_next()

feature_spec = {"X": tf.placeholder(dtype=tf.float64, shape=[None] + shape)}
#serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn({"X": tf.FixedLenFeature(shape=shape, dtype=tf.string)})
serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)

folder = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
out_dir = os.path.join("data", folder)
model_params ={"learning_rate": 0.01, "out_dir": out_dir}

config = tf.estimator.RunConfig(model_dir=out_dir, save_summary_steps=100, tf_random_seed=42)

nn = tf.estimator.Estimator(model_fn=model_fn, params=model_params, config=config)

#print(nn.get_variable_names())

print(ops.Graph().get_operations())

#print(tf.get_default_graph().get_tensor_by_name("dense/layer0:0"))

hooks = [tf.train.StopAtStepHook(num_steps=2000)]

nn.train(input_fn=train_input_fn, hooks=hooks)

nn.export_savedmodel(out_dir, serving_input_receiver_fn)
