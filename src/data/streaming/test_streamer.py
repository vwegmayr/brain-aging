from modules.models.data_transform import DataTransformer
import tensorflow as tf
import numpy as np


class TestStreamer(DataTransformer):
    def __init__(self, streamer):
        _class = streamer["class"]
        self.streamer = _class(**streamer["params"])
        self.sess = tf.Session()

    def transform(self, X, y=None):
        # Stream all images and check if they are None
        fn = self.streamer.get_input_fn(train=True)

        e = fn()
        for i in range(2):
            print(i)
            r = self.sess.run(e)
            print("image_label = {}".format(r["patient_label_0"]))
            print(r["X_0"].shape)
            for k in r:
                for v in r[k]:
                    assert v is not None

        self.streamer = None
        self.sess = None
