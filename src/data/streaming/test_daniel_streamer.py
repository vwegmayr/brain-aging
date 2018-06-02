from modules.models.data_transform import DataTransformer
import tensorflow as tf
import numpy as np

import src.features as ft_def


class TestStreamer(DataTransformer):
    def __init__(self, streamer):
        _class = streamer["class"]
        self.streamer = _class(**streamer["params"])
        self.sess = tf.Session()

        ft_def.all_features.feature_info[ft_def.MRI]['shape'] = \
            self.streamer.get_mri_shape()

    def transform(self, X, y=None):
        # Stream all images and check if they are None
        fn = self.streamer.get_input_fn(train=True)

        e = fn()
        for i in range(1):
            print(i)
            r = self.sess.run(e)
            print(r["mri"][0].shape)
            for k in r:
                if k != "mri":
                    print(r[k][0])
            """
            # print("image_label = {}".format(r["patient_label_0"]))
            #print(r["X_0"].shape)
            for k in r:
                for v in r[k]:
                    assert v is not None
            """
        self.streamer = None
        self.sess = None
