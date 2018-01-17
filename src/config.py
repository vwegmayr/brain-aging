import tensorflow as tf

prefix_data = '/local/home/dhaziza/entrack/data/'
prefix_data_raw = prefix_data + 'raw/'
prefix_data_converted = prefix_data + 'ready/'
train_database_file = 'train.tfrecord'
test_database_file = 'test.tfrecord'
image_shape = (256, 256, 128)
dataset_compression = tf.python_io.TFRecordCompressionType.GZIP
test_set_size_ratio = 0.2
test_set_random_seed = 0
