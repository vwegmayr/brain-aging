from modules.models.data_transform import DataTransformer
import tensorflow as tf
import numpy as np


class TestStreamer(DataTransformer):
    def __init__(self, streamer):
        _class = streamer["class"]
        self.streamer = _class(**streamer["params"])
        self.sess = tf.Session()

    def transform(self, X, y=None):
        print("Sample shape {}".format(self.streamer.get_sample_shape()))
        for group in self.streamer.groups[:10]:
            print(group)
            fid = group.file_ids[0]
            p = self.streamer.get_file_path(fid)
            # print(self.streamer.load_sample(p))

        self.streamer = None
        self.sess = None
