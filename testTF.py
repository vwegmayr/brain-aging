import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

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

def model_fn(features, labels, mode, params):
  """Model function for Estimator."""

  # Connect the first hidden layer to input layer
  # (features["x"]) with relu activation
  first_hidden_layer = tf.layers.dense(features["X"], 10, activation=tf.nn.relu)

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
        predictions={"ages": predictions})

  # Calculate loss using mean squared error
  loss = tf.losses.mean_squared_error(labels, predictions)

  # Calculate root mean squared error as additional eval metric
  eval_metric_ops = {
      "rmse": tf.metrics.root_mean_squared_error(
          tf.cast(labels, tf.float64), predictions)
  }

  optimizer = tf.train.GradientDescentOptimizer(
      learning_rate=params["learning_rate"])
  train_op = optimizer.minimize(
      loss=loss, global_step=tf.train.get_global_step())

  # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops)

def train_input_fn():
  dataset = tf.data.TFRecordDataset(["data/tf.test"])
  dataset = dataset.map(parser)
  dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.batch(2)
  dataset = dataset.repeat(1)
  iterator = dataset.make_one_shot_iterator()

  return iterator.get_next()

model_params ={"learning_rate": 0.01}

config = tf.estimator.RunConfig(model_dir=".")

nn = tf.estimator.Estimator(model_fn=model_fn, params=model_params, config=config)

nn.train(input_fn=train_input_fn)