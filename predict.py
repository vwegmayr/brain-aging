import tensorflow as tf
#from testTF import model_fn
import numpy as np
from google.protobuf.json_format import MessageToJson

"""
out_dir="data/2017-11-20_21:52:46"

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

def test_input_fn():
  dataset = tf.data.TFRecordDataset(["data/tf.test"])
  dataset = dataset.map(parser)
  dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.batch(100)
  dataset = dataset.repeat(1)
  iterator = dataset.make_one_shot_iterator()

  return iterator.get_next()

config = tf.estimator.RunConfig(model_dir=out_dir)
nn = tf.estimator.Estimator(model_fn=model_fn, config=config)

for i, y in enumerate(nn.predict(input_fn=test_input_fn)):
	if i < 5:
		print(y)
"""

"""
export_dir="data/2017-11-22_14:20:12/1511360413"

predict = tf.contrib.predictor.from_saved_model(export_dir)

print(predict({"X": np.random.rand(3,20)}))
"""

iterator = tf.python_io.tf_record_iterator("data/tf.test")
print(MessageToJson(tf.train.Example.FromString(next(iterator))))