import numpy as np
import tensorflow as tf
import sklearn as skl

from modules.models.utils import parse_hooks, parse_layers
from modules.models.base import DeterministicTracker, ProbabilisticTracker

from tensorflow.python.saved_model.signature_constants import (
    DEFAULT_SERVING_SIGNATURE_DEF_KEY)
from tensorflow.python.estimator.export.export_output import PredictOutput


class SimpleTracker(DeterministicTracker):
    """docstring for ExampleTF"""

    def __init__(self, input_fn_config={"shuffle": True}, config={},
                 params={}):  # noqa: E129

        super(SimpleTracker, self).__init__(input_fn_config, config, params)

    def model_fn(self, features, labels, mode, params, config):

        blocks = tf.layers.flatten(features["blocks"])

        incoming = tf.layers.flatten(features["incoming"])

        concat = tf.concat([blocks, incoming], axis=1)

        unnormed = parse_layers(
            inputs=concat,
            layers=params["layers"],
            mode=mode,
            default_summaries=params["default_summaries"])

        normed = tf.nn.l2_normalize(unnormed, dim=1)

        predictions = normed
        # ================================================================
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"predictions": predictions},
                export_outputs={
                    DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    PredictOutput({
                        "predictions": predictions
                    })
                })
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
            training_hooks = parse_hooks(params["hooks"], locals(),
                                         self.save_path)
        else:
            training_hooks = []
        # ================================================================
        if (mode == tf.estimator.ModeKeys.TRAIN
                or mode == tf.estimator.ModeKeys.EVAL):  # noqa: E129
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                training_hooks=training_hooks)

    def score(self, X):
        pass


class MaxEntropyTracker(ProbabilisticTracker):
    """Implementation of the maximimum entropy probabilistic tracking."""

    def __init__(self, input_fn_config={"shuffle": True}, config={},
                 params={}):  # noqa: E129

        super(MaxEntropyTracker, self).__init__(input_fn_config, config,
                                                params)

    def model_fn(self, features, labels, mode, params, config):

        blocks = tf.layers.flatten(features["blocks"])

        incoming = tf.layers.flatten(features["incoming"])

        concat = tf.concat([blocks, incoming], axis=1)

        last_layer = parse_layers(
            inputs=concat,
            layers=params["layers"],
            mode=mode,
            default_summaries=params["default_summaries"])

        # After the last layer specified, add just 1 weight layer to the mean
        # vectors and one to the concentration values.

        # TODO: Make this modular. Parameters should be passed in the config
        # file.
        key = "dense"
        mu_params = {'activation': tf.nn.relu, 'units': 512}
        with var_scope("last_mean", values=(unnormed,)) as scope:
            mu_out = getattr(tf.layers, key)(
                inputs, **mu_params, name=scope)

        k_params = {'activation': tf.nn.relu, 'units': 512}
        with var_scope("last_k", values=(unnormed,)) as scope:
            k_out = getattr(tf.layers, key)(
                inputs, **k_params, name=scope)

        if default_summaries is not None:
            for summary in default_summaries:
                summary["sum_op"](name, inputs)

        # Normalize the mean vectors
        mu_normed = tf.nn.l2_normalize(mu_out, dim=1)

        # TODO: How to pass out more predictions than 1? Dictionary??
        # In base.ProbabilisticTracker the implementation already includes
        # a dictionary with 'mean' and 'concentration' keys. Sticking to that
        # for the moment.
        predictions = {
            'mean': mu_normed,       # Complying with base.ProbabilisticTracker
            'concentration': k_out   # Complying with base.ProbabilisticTracker
        }
        # ================================================================
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"predictions": predictions},
                export_outputs={
                    DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    PredictOutput({
                        "predictions": predictions
                    })
                })
        # ================================================================

        # TODO: Introduce temperature parameter T in the config file.
        cur_T = 1
        loss = self.max_entropy_loss(y=labels, mu=mu_normed, k=k_out, T=cur_T)

        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=params["learning_rate"])

        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        # ================================================================
        if "hooks" in params:
            training_hooks = parse_hooks(params["hooks"], locals(),
                                         self.save_path)
        else:
            training_hooks = []
        # ================================================================
        if (mode == tf.estimator.ModeKeys.TRAIN
                or mode == tf.estimator.ModeKeys.EVAL):  # noqa: E129
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                training_hooks=training_hooks)

    def score(self, X):
        pass

    @staticmethod
    def max_entropy_loss(y, mu, k, T):
        """Compute the maximum entropy loss.

        Args:
            y: Ground-truth fiber direction vectors.
            mu: Predicted mean vectors.
            k: Concentration parameters.
            T: Temperature parameter.

        Returns:
            loss: The maximum entropy loss.

        """
        dot_products = tf.reduce_sum(tf.multiply(mu, y), axis=1)
        cost = -tf.multiply(
            (tf.cosh(k) / tf.sinh(k) - tf.reciprocal(k)), dot_products)
        entropy = 1 - k / tf.tanh(k) - tf.log(k / (4 * np.pi * tf.sinh(k)))
        loss = cost - T * entropy
        loss = tf.reduce_mean(loss)
        return loss
