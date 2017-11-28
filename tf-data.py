import nibabel as nib
import tensorflow as tf
import glob
import os
import numpy as np
import pandas as pd
import argparse

def tf_feature(value, list_type):
  if not isinstance(value, list):
    value = [value]

  if list_type not in ["Int64", "Float", "Bytes"]:
    raise RuntimeError("Unsupported list_type: {}".format(list_type))

  tf_list = getattr(tf.train, list_type + "List")(value=value)
  tf_dict = {list_type.lower() + "_list": tf_list}

  return tf.train.Feature(**tf_dict)


def transform_files_to_tfrecord(args):
  writer = tf.python_io.TFRecordWriter(args.output)

  if args.type == "nii":

    labels = pd.read_csv(args.labels)
    labels = labels.set_index("file").to_dict(orient="index")


    for file in glob.glob(os.path.join(args.input, "*.nii*")):
      image = nib.load(file).get_data()
      feature = {"image": tf_feature(image.tostring(), "Bytes"),
                 "shape": tf_feature(list(image.shape), "Int64"),
                 "label": tf_feature(labels[file]["label"], "Int64")}
      example = tf.train.Example(features=tf.train.Features(feature=feature))
      writer.write(example.SerializeToString())
    
  else:
    np.random.seed(42)
    labels = np.random.randint(2, size=100)
    data = np.random.rand(100,20)
    for i in range(len(labels)):
      feature = {"X": tf_feature(data[i, :].tostring(), "Bytes"),
                 "y": tf_feature(labels[i], "Int64"),
                 "X_shape": tf_feature([20], "Int64"),
                 "X_dtype": tf_feature([b for b in "float64".encode("ascii")], "Int64")}
      example = tf.train.Example(features=tf.train.Features(feature=feature))
      writer.write(example.SerializeToString())

  writer.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Transform set of files to TFRecord binary file.")
  parser.add_argument("-i", "--input", help="Path to directory containing the files.")
  parser.add_argument("-L", "--labels", help="Path to labels.")
  parser.add_argument("-o", "--output", help="Name of output TFRecord file.")
  parser.add_argument("-t", "--type", help="File format of input", choices=["nii"])
  args = parser.parse_args()

  transform_files_to_tfrecord(args)