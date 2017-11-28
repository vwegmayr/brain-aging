import tensorflow as tf
from abc import ABC, abstractmethod
from .parsing import get_example_structure

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


def get_input_fns(data_dict):
  input_fns = {"train": None, "eval": None, "test": None}

  for mode in input_fns.keys():
    if mode in data_dict:
      example_structure = get_example_structure(data_dict[mode]["files"][0])
      parser = data_dict[mode]["parse_fn"](example_structure)
      input_fns[mode] = Input(tf_files=data_dict[mode]["files"], parser=parser.parse_fn).input_fn

  return input_fns