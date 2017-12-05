import numpy as np
import tensorflow as tf
import sklearn as skl

from sklearn.utils.validation import check_array, check_is_fitted

from modules.models.utils import parse_hooks
from modules.models.base import BaseTF

from tensorflow.python.saved_model.signature_constants import (
    DEFAULT_SERVING_SIGNATURE_DEF_KEY)
from tensorflow.python.estimator.export.export_output import PredictOutput


class BasicTracker(BaseTF):
    """docstring for ExampleTF"""
    def __init__(
        self,
        input_fn_config={"shuffle": True},
        config={},
        params={}):  # noqa: E129

        super(ExampleTF, self).__init__(input_fn_config, config, params)

    def model_fn(self, features, labels, mode, params, config):



        input_tensor = tf.cast(features["blocks"], tf.float32)
        input_tensor = tf.expand_dims(input_tensor, axis=-1)

        conv_layer = tf.layers.conv1d(
            input_tensor,
            filters=16,
            kernel_size=256,
            strides=64,
            activation=tf.nn.relu)

        max_pooled = tf.layers.max_pooling1d(
            conv_layer,
            pool_size=64,
            strides=1)

        flat = tf.layers.flatten(max_pooled)

        dense_layer_1 = tf.layers.dense(
            flat,
            units=512,
            activation=tf.nn.relu)

        dense_layer_1_norm = tf.norm(dense_layer_1)

        logits = tf.layers.dense(
            dense_layer_1,
            units=4,
            activation=None)

        probabs = tf.nn.softmax(logits)

        predictions = tf.argmax(probabs, axis=1)
        # ================================================================
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={
                    "predictions": predictions,
                    "probabs": probabs},
                export_outputs={
                    DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    PredictOutput({"predictions": predictions}),
                    "probabs":
                    PredictOutput({"probabs": probabs})})
        # ================================================================
        labels = tf.cast(labels, tf.int32)
        labels = tf.one_hot(labels, depth=4)

        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits)
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

    def score(self, X, y):
        y_pred = self.predict(X)
        return skl.metrics.f1_score(y, y_pred, average="micro")
