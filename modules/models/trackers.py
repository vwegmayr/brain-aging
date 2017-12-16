import numpy as np
import tensorflow as tf
import sklearn as skl

from modules.models.utils import parse_hooks, parse_layers
from modules.models.base import DeterministicTracker

from tensorflow.python.saved_model.signature_constants import (
    DEFAULT_SERVING_SIGNATURE_DEF_KEY)
from tensorflow.python.estimator.export.export_output import PredictOutput


class SimpleTracker(DeterministicTracker):
    """docstring for ExampleTF"""
    def __init__(
        self,
        input_fn_config={"shuffle": True},
        config={},
        params={}):  # noqa: E129

        super(SimpleTracker, self).__init__(
            input_fn_config,
            config,
            params)

    def model_fn(self, features, labels, mode, params, config):

        blocks = tf.layers.flatten(features["blocks"])

        incoming = tf.layers.flatten(features["incoming"])

        concat = tf.concat([blocks, incoming], axis=1)

        unnormed = parse_layers(
            inputs=concat,
            layers=params["layers"],
            mode=mode,
            default_summaries=params["default_summaries"]
        )

        normed = tf.nn.l2_normalize(unnormed, dim=1)

        predictions = normed
        # ================================================================
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={
                    "predictions": predictions},
                export_outputs={
                    DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    PredictOutput({"predictions": predictions})})
        # ================================================================
        loss = -tf.multiply(normed, labels)
        loss = tf.reduce_sum(loss, axis=1)
        loss = tf.reduce_mean(loss)

        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=params["learning_rate"])

        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        # ================================================================
        if "hooks" in params:
            training_hooks = parse_hooks(
                params["hooks"],
                locals(),
                self.save_path)
        else:
            training_hooks = []
        # ================================================================
        if (mode == tf.estimator.ModeKeys.TRAIN or
            mode == tf.estimator.ModeKeys.EVAL):  # noqa: E129
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                training_hooks=training_hooks)

    def score(self, X):
        pass
