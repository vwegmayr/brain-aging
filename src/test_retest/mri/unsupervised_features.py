from radiomics import featureextractor
import SimpleITK as sitk
import os
import numpy as np
import json
from memory_profiler import profile
import gc
import tensorflow as tf

from modules.models.data_transform import DataTransformer
from src.test_retest.test_retest_base import EvaluateEpochsBaseTF
from src.test_retest.test_retest_base import linear_trafo
from src.test_retest.test_retest_base import regularizer


class PyRadiomicsFeatures(DataTransformer):
    def __init__(self, streamer):
        # Initialize streamer
        _class = streamer["class"]
        self.streamer = _class(**streamer["params"])

    def get_extractor(self):
        # Initialize extractor
        extractor = featureextractor.RadiomicsFeaturesExtractor()
        extractor.enableAllImageTypes()
        extractor.enableAllFeatures()

        return extractor

    def transform(self, X, y=None):
        out_path = os.path.join(self.save_path, "features")
        os.mkdir(out_path)
        # Stream image one by one
        batches = self.streamer.get_batches()
        for batch in batches:
            for group in batch:
                for file_id in group.get_file_ids():
                    path = self.streamer.get_file_path(file_id)

                    sitk_im = sitk.ReadImage(path)
                    all_ones = np.ones(sitk_im.GetSize())
                    sitk_mask = sitk.GetImageFromArray(all_ones)

                    extractor = self.get_extractor()
                    features = extractor.computeFeatures(sitk_im, sitk_mask, "brain")

                    with open(
                        os.path.join(out_path, str(file_id) + ".json"),
                        "w"
                    ) as f:
                        json.dump(features, f, indent=2)


class LinearAutoEncoder(EvaluateEpochsBaseTF):
    def model_fn(self, features, labels, mode, params):
        input_mri = tf.reshape(
            features["mri"],
            [-1, params["input_dim"]]
        )

        hidden_dim = params["input_dim"]
        w, b, hidden = linear_trafo(
            X=input_mri,
            out_dim=hidden_dim,
            types=[tf.float16, tf.float16],
            names=["weights", "bias", "hidden_rep"]
        )

        reconstruction = tf.add(
            tf.matmul(hidden, tf.transpose(w)),
            b,
            name="reconstruction"
        )

        predictions = {
            "hidden_rep": hidden,
            "reconstruction": reconstruction
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions
            )

        # Compute loss
        loss = tf.reduce_sum(tf.square(input_mri - reconstruction))

        optimizer = tf.train.AdamOptimizer(
            learning_rate=params["learning_rate"]
        )
        train_op = optimizer.minimize(loss, tf.train.get_global_step())

        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op
            )

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss
        )

    def gen_input_fn(self, X, y=None, train=True, input_fn_config={}):
        return self.streamer.get_input_fn(train)
