import numpy as np
import tensorflow as tf
import sklearn as skl
import scipy as sp
import os

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ml_project.models.utils import parse_hooks
from ml_project.models.base import BaseTF

class MeanPredictor(BaseEstimator, TransformerMixin):
    """docstring for MeanPredictor"""

    def fit(self, X, y):
        self.mean = y.mean(axis=0)
        return self

    def predict_proba(self, X):
        check_array(X)
        check_is_fitted(self, ["mean"])
        n_samples, _ = X.shape
        return np.tile(self.mean, (n_samples, 1))


class ExampleTF(BaseTF):
    """docstring for ExampleTF"""
    def __init__(self, input_fn_config={"shuffle": True}, config={}, params={}):
        super(ExampleTF, self).__init__(input_fn_config, config, params)

    def model_fn(self, features, labels, mode, params, config):
        #================================================================

        input_tensor = tf.cast(features["X"], tf.float32)
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

        dense_layer_1 = tf.layers.dense(
            max_pooled,
            units=512,
            activation=tf.nn.relu)

        logits = tf.layers.dense(
            dense_layer_1,
            units=4,
            activation=None)

        probabs = tf.nn.softmax(logits)
        predictions = tf.nn.argmax(probabs)

        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(labels, depth=4),
            logits=logits)

        """
        first_hidden_layer = tf.layers.dense(features["X"], 10, activation=tf.nn.relu, name="layer0")

        second_hidden_layer = tf.layers.dense(first_hidden_layer, 10, activation=tf.nn.relu)

        output_layer = tf.layers.dense(second_hidden_layer, 1)

        mean_output = tf.reduce_mean(output_layer)

        predictions = tf.reshape(output_layer, [-1])
        """

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={
                    "predictions": predictions,
                    "probabs": probabs},
                export_outputs={
                "predictions": tf.estimator.export.PredictOutput({"predictions": predictions}),
                "probabs": tf.estimator.export.PredictOutput({"probabs": probabs})
                })

        #loss = tf.losses.mean_squared_error(labels, predictions)

        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=params["learning_rate"])

        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())

        #================================================================
        #eval_metric_ops = {
        #    "score": tf.metrics.root_mean_squared_error(
        #        tf.cast(labels, tf.float64), predictions)
        #}

        #================================================================
        if "hooks" in params:
            training_hooks = parse_hooks(params["hooks"], locals(), self.save_path)
        else:
            training_hooks = []

        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
          return tf.estimator.EstimatorSpec(
              mode=mode,
              loss=loss,
              train_op=train_op,
              eval_metric_ops=eval_metric_ops,
              training_hooks=training_hooks)

    def score(self, X, y):
        y_pred = self.predict(X)
        return skl.metrics.f1_score(y, y_pred, average="micro")