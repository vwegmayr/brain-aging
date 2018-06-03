from modules.models.data_transform import DataTransformer
import tensorflow as tf
import numpy as np


class TestStreamer(DataTransformer):
    def __init__(self, streamer):
        _class = streamer["class"]
        self.streamer = _class(**streamer["params"])
        self.sess = tf.Session()

    def transform(self, X, y=None):
        for group in self.streamer.groups[:10]:
            print(group)

        self.streamer = None
        self.sess = None
