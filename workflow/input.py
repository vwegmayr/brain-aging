import tensorflow as tf
from abc import ABC, abstractmethod


class Input(object):
    """docstring for ClassName"""
    def __init__(self,
                 tf_files=[],
                 parser=None,
                 shuffle=False,
                 buffer_size=1,
                 batch_size=1,
                 epochs=1):

        super(Input, self).__init__()

        self.tf_files = tf_files
        self.parser = parser
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epochs = epochs

    def input_fn(self):
        dataset = tf.data.TFRecordDataset(self.tf_files)
        dataset = dataset.map(self.parser)
        if self.shuffle:
            dataset = dataset.shuffle(self.buffer_size)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat(self.epochs)

        iterator = dataset.make_one_shot_iterator()

        return iterator.get_next()

