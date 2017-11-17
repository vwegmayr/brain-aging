import nibabel as nib
import tensorflow as tf
import glob
import os
import numpy as np

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

image_folder = "/local/OASIS/OASIS_normalized/CV1/*.nii.gz"

writer = tf.python_io.TFRecordWriter("data/tf.test")

num_samples = len(glob.glob(image_folder))
print("num samples = {}".format(num_samples))

np.random.seed(42)

labels = np.random.randint(2, size=num_samples)

i=0
for file in glob.glob(image_folder):
    img = nib.load(file)
    shape = img.shape
    #data = img.get_data()
    data = np.random.rand(20)

    feature = {
               "y": _int64_feature(labels[i]),
               "X": _bytes_feature(data.tostring())
              }


    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())
    i += 1

writer.close()