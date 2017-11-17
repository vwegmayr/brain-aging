import numpy as np
import tensorflow as tf
import scipy as sp
import os
from ml_project.models import utils, layers
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from abc import ABC, abstractmethod


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


class BaseTF(ABC, BaseEstimator):
    """docstring for BaseTF"""

    def __init__(self, batch_size, num_epochs, gradient_max_norm, optimizer_conf,
                 loss_function, layers, random_seed):
        super(BaseTF, self).__init__()

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.gradient_max_norm = gradient_max_norm
        self.optimizer_conf = optimizer_conf
        self.loss_function = loss_function["class"]
        self.layers = layers
        self.random_seed = random_seed

        self._sess_set = False
        self._optimizer_set = False
        self._graph_set = False
        self._checkpoint_path_set = False
        self._model_set = False

        tf.set_random_seed(self.random_seed)

    def set_save_path(self, save_path):
        self.save_path = save_path
        if not self._checkpoint_path_set and save_path is not None:
            self.checkpoint_path = save_path + self.__class__.__name__ + ".ckpt"
            self._checkpoint_path_set = True

    def check_shape_X(self, X):
        if list(X.shape[1:]) != self.input_shape:
            raise RuntimeError("Input shape of X does not match, "
                               "got {}, expected {}.".format(list(X.shape[1:]), self.input_shape))

    def check_shape_X_y(self, X, y):
        check_shape_X(X)

        if list(y.shape[1:]) != self.output_shape:
            raise RuntimeError("Input shape of y does not match, "
                               "got {}, expected {}.".format(list(y.shape[1:]), self.output_shape))

    @abstractmethod
    def parser(self, record):
        pass

    @abstractmethod
    def _partial_fit(self, X, y):
        pass

    @abstractmethod
    def _set_graph(self):
        pass

    def _set_model(self):
        if not self._model_set:
            self._set_optimizer()
            self._set_graph()
            self._set_sess()

            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            self._sess.run(init_op)
            self._saver = tf.train.Saver()

            self._model_set = True

    def _restore_model(self):
        if not self._model_set:
            self._set_sess()
            self._set_optimizer()
            self._set_graph()

            self._saver = tf.train.Saver()

            if os.path.exists(self.checkpoint_path + ".meta"):
                self._saver.restore(self._sess, self.checkpoint_path)
            else:
                raise RuntimeError("Can not retrieve checkpoint file {}, "
                                   "no such file.".format(self.checkpoint_path))

            self._model_set = True


    def _set_sess(self):
        if not self._sess_set:
            self._sess = tf.Session()
            self._sess_set = True

    def _set_optimizer(self):
        if not self._optimizer_set:
            self.optimizer = utils.get_object(self.optimizer_conf["module"],
                                              self.optimizer_conf["class"])(**self.optimizer_conf["params"])
            self._optimizer_set = True

    def __getstate__(self):
        state = self.__dict__.copy()

        for key, val in list(state.items()):
            if "tensorflow" in getattr(val, "__module__", "None"):
                del state[key]

        state["_sess_set"] = False
        state["_optimizer_set"] = False
        state["_graph_set"] = False
        state["_model_set"] = False
        
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class NeuralNetwork(BaseTF):
    """docstring for NeuralNetwork"""

    def __init__(self, layers=None, batch_size=1, num_epochs=1,
                 optimizer_conf=None, gradient_max_norm=None,
                 loss_function=None, random_seed=42):
        super(NeuralNetwork, self).__init__(batch_size, num_epochs, gradient_max_norm,
                                            optimizer_conf, loss_function, layers,
                                            random_seed)

    def fit(self, data, y=None):
        
        self._set_model()
        #self._sess.run(self.train_op)

        self._sess.run(self.iterator.initializer, feed_dict={self.filenames: ["data/tf.test"]})
        
        while True:
            try:
                self._sess.run(self.train_op)
            except tf.errors.OutOfRangeError:
                break
        
        """
        batches_per_epoch = int(data.num_samples("train") / self.batch_size)
        self.input_shape = data.input_shape()
        self.output_shape = data.output_shape()
        self._set_model()
        for epoch in range(self.num_epochs):
            for batch in range(batches_per_epoch):
                X_batch, y_batch = data.get_batch(self.batch_size, "train")
                self._partial_fit(X_batch, y_batch)
        """
        if self.save_path is not None:
            self._saver.save(self._sess, self.save_path + self.__class__.__name__ + ".ckpt")
        
        return self

    def _partial_fit(self, X, y):
        X, y = check_X_y(X, y, force_all_finite=True, allow_nd=True, multi_output=True)
        loss, _ = self._sess.run([self.loss, self.train_op],
                                 feed_dict={self._X: X, self._y: y, self._training: True})
        return loss

    def predict(self, data):
        X = data.X
        X = check_array(X, force_all_finite=True, allow_nd=True)
        check_is_fitted(self, ["input_shape", "output_shape"])
        self.check_shape_X(X)
        self._restore_model()
        logits = self._sess.run([self.logits],
                                 feed_dict={self._X: X, self._training: False})

        return utils.logits2labels(logits)

    def predict_proba(self, X):
        X = check_array(X, force_all_finite=True, allow_nd=True)
        check_is_fitted(self, ["input_shape", "output_shape"])
        self.check_shape_X(X)
        self._restore_model()
        logits = self._sess.run([self.logits],
                                 feed_dict={self._X: X, self._training: False})

        return utils.logits2proba(logits)

    def score(self, X, y):
        X, y = check_X_y(X, y, force_all_finite=True, allow_nd=True, multi_output=True)
        check_is_fitted(self, ["input_shape", "output_shape"])
        self.check_shape_X_y(X, y)
        self._restore_model()
        loss = self._sess.run([self.loss],
                                 feed_dict={self._X: X, self._y: y, self._training: False})
        return loss

    def parser(self, record):
       keys_to_features = {
           "X": tf.FixedLenFeature(shape=[], dtype=tf.string),
           "y": tf.FixedLenFeature(shape=[], dtype=tf.int64),
       }
       parsed = tf.parse_single_example(record, features=keys_to_features)

       image = tf.decode_raw(parsed["X"], tf.float64)
       #image.set_shape([20])
       image = tf.reshape(image, [20])
       label = tf.cast(parsed["y"], tf.float64)
       #label.set_shape([])
       label = tf.reshape(label, [1])

       return image, label

    def _set_graph(self):
        if not self._graph_set:
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            self._training = tf.placeholder_with_default(True, shape=[], name="training")

            self.filenames = tf.placeholder(tf.string, shape=[None])
            
            self.dataset = tf.data.TFRecordDataset(self.filenames)
            self.dataset = self.dataset.map(self.parser)
            self.dataset = self.dataset.shuffle(buffer_size=10000)
            self.dataset = self.dataset.batch(self.batch_size)
            self.dataset = self.dataset.repeat(self.num_epochs)
            self.iterator = self.dataset.make_initializable_iterator()
        
            self._X, self._y = self.iterator.get_next()
            """
            self._X = tf.placeholder(
                tf.float32, [None] + utils.make_list(self.input_shape), name="X")
            self._y = tf.placeholder(
                tf.float32, [None] + utils.make_list(self.output_shape), name="y")
            """
            
            self.logits = tf.contrib.layers.flatten(
            layers.build_architecture(x=self._X, architecture=self.layers, scope="layers",
                training=self._training))

            self.loss = self.loss_function(self.logits, self._y)

            gradients, variables = zip(
                *self.optimizer.compute_gradients(self.loss))
            if self.gradient_max_norm is not None:
                gradients, _ = tf.clip_by_global_norm(
                    gradients, self.gradient_max_norm)

            self.train_op = self.optimizer.apply_gradients(
                zip(gradients, variables), global_step=self.global_step, name="train_op")
            
            self._graph_set = True


def mean_cross_entropy(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))