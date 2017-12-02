import tensorflow as tf
import multiprocessing
import os

from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
from ml_project.models.utils import print
from tensorflow.estimator.export import (
    build_raw_serving_input_receiver_fn as input_receiver_fn)


class BaseTF(ABC, BaseEstimator, TransformerMixin):
    """docstring for BaseTF"""
    lock = multiprocessing.Lock()
    num_instances = 0

    def __init__(self, input_fn_config, config, params):
        super(BaseTF, self).__init__()
        self.input_fn_config = input_fn_config
        self.config = config
        self.params = params
        self.id = id

        self._restore_path = None

        with BaseTF.lock:
            self.instance_id = BaseTF.num_instances
            BaseTF.num_instances += 1

    def fit(self, X, y):
        with BaseTF.lock:
            config = self.config
            if BaseTF.num_instances > 1:
                config["model_dir"] = os.path.join(
                    config["model_dir"],
                    "inst-" + str(self.instance_id))

        self.estimator = tf.estimator.Estimator(
            model_fn=self.model_fn,
            params=self.params,
            config=tf.estimator.RunConfig(**config))

        tf.logging.set_verbosity(tf.logging.INFO)
        try:
            self.estimator.train(input_fn=self.input_fn(X, y))
        except KeyboardInterrupt:
            print("\nEarly stop of training, saving model...")
            self.export_estimator(
                input_shape=list(X.shape[1:]),
                input_dtype=X.dtype.name)
        else:
            self.export_estimator(
                input_shape=list(X.shape[1:]),
                input_dtype=X.dtype.name)

        return self

    def predict(self, X, head="predictions"):
        predictor = tf.contrib.predictor.from_saved_model(self._restore_path)
        return predictor({"X": X})[head]

    def predict_proba(self, X):
        return self.predict(X, head="probabs")

    def input_fn(self, X, y):
        return tf.estimator.inputs.numpy_input_fn(
            x={"X": X},
            y=y,
            **self.input_fn_config)

    def set_save_path(self, save_path):
        self.save_path = save_path
        if self._restore_path is None:
            self.config["model_dir"] = save_path

    def export_estimator(self, input_shape, input_dtype):
        feature_spec = {"X": tf.placeholder(
            shape=[None] + input_shape,
            dtype=input_dtype)}
        receiver_fn = input_receiver_fn(feature_spec)
        self._restore_path = self.estimator.export_savedmodel(
            self.save_path,
            receiver_fn)
        print("Model saved to {}".format(self._restore_path))

    @abstractmethod
    def score(self, X, y):
        pass

    @abstractmethod
    def model_fn(self, features, labels, mode, params, config):
        pass

    def __getstate__(self):
        state = self.__dict__.copy()

        for key, val in list(state.items()):
            if "tensorflow" in getattr(val, "__module__", "None"):
                del state[key]

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
