import sklearn as skl
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from abc import ABC, abstractmethod
from workflow.model_functions import parse_hooks
import os
import multiprocessing
from .utils import print


class KernelEstimator(skl.base.BaseEstimator, skl.base.TransformerMixin):
    """docstring"""
    def __init__(self, save_path=None):
        super(KernelEstimator, self).__init__()
        self.save_path = save_path

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.y_mean = np.mean(y)
        y -= self.y_mean
        Xt = np.transpose(X)
        cov = np.dot(X, Xt)
        alpha, _, _, _ = np.linalg.lstsq(cov, y)
        self.coef_ = np.dot(Xt, alpha)

        if self.save_path is not None:
            plt.figure()
            plt.hist(self.coef_[np.where(self.coef_ != 0)], bins=50,
                     normed=True)
            plt.savefig(self.save_path + "KernelEstimatorCoef.png")
            plt.close()

        return self

    def predict(self, X):
        check_is_fitted(self, ["coef_", "y_mean"])
        X = check_array(X)

        prediction = np.dot(X, self.coef_) + self.y_mean

        if self.save_path is not None:
            plt.figure()
            plt.plot(prediction, "o")
            plt.savefig(self.save_path + "KernelEstimatorPrediction.png")
            plt.close()

        return prediction

    def score(self, X, y, sample_weight=None):
        scores = (self.predict(X) - y)**2 / len(y)
        score = np.sum(scores)

        if self.save_path is not None:
            plt.figure()
            plt.plot(scores, "o")
            plt.savefig(self.save_path + "KernelEstimatorScore.png")
            plt.close()

            df = pd.DataFrame({"score": scores})
            df.to_csv(self.save_path + "KernelEstimatorScore.csv")

        return score

    def set_save_path(self, save_path):
        self.save_path = save_path


class BaseTF(ABC, BaseEstimator):
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

        try:
            self.estimator.train(input_fn=self.input_fn(X, y))
            self.export_estimator(
                input_shape=list(X.shape[1:]),
                input_dtype=X.dtype.name)
        except Exception as err:
            print(err)

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
        feature_spec = {"X": tf.placeholder(shape=[None] + input_shape, dtype=input_dtype)}
        receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
        self._restore_path = self.estimator.export_savedmodel(self.save_path, receiver_fn)
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


class ExampleTF(BaseTF):
    """docstring for ExampleTF"""
    def __init__(self, input_fn_config={"shuffle": True}, config={}, params={}):
        super(ExampleTF, self).__init__(input_fn_config, config, params)

    def model_fn(self, features, labels, mode, params, config):
        #================================================================
        first_hidden_layer = tf.layers.dense(features["X"], 10, activation=tf.nn.relu, name="layer0")

        second_hidden_layer = tf.layers.dense(first_hidden_layer, 10, activation=tf.nn.relu)

        output_layer = tf.layers.dense(second_hidden_layer, 1)

        mean_output = tf.reduce_mean(output_layer)

        predictions = tf.reshape(output_layer, [-1])

        if mode == tf.estimator.ModeKeys.PREDICT:
          return tf.estimator.EstimatorSpec(
              mode=mode,
              predictions={"predictions": predictions},
              export_outputs={"predictions": tf.estimator.export.PredictOutput({"predictions": predictions})})

        loss = tf.losses.mean_squared_error(labels, predictions)

        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=params["learning_rate"])

        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())

        #================================================================
        eval_metric_ops = {
            "score": tf.metrics.root_mean_squared_error(
                tf.cast(labels, tf.float64), predictions)
        }

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
        return skl.metrics.mean_squared_error(y, y_pred)