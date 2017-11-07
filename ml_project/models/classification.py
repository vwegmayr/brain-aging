import numpy as np
import tensorflow as tf
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

    def __init__(self):
        super(BaseTF, self).__init__()
        self._graph_built = False
        self._saver = None

    def _init_session(self, tf_config=None):
        if self.optimizer is None:
            raise RuntimeError(
                "Optimizer not set for {}".format(self.__name__))

        if not self._graph_built:
            self._sess = tf.Session(config=tf_config)
            # Init the network parameters and build the computational graph
            self._build_graph()

            # Merge all the summaries and dump them to the disk
            # self._summary_merged = tf.summary.merge_all()
            # self._summary_writer = tf.summary.FileWriter(self.summary_folder, self._sess.graph)

            # Init Tensorflow variables
            init = tf.global_variables_initializer()
            self._sess.run(init)

            # Tensorflow Saver object
            self._saver = tf.train.Saver()
            self._graph_built = True

    def restore_checkpoint(self, checkpoint_path=None):
        """
        Restores a trained model. If the argument :code:`checkpoint_path` is not :code:`None`, then
        the model is restored from this file. Otherwise the checkpoint path provided by the
        initializer argument :code:`checkpoint_path` is used.
        
        Args:
            checkpoint_path (string): Optional, path to a checkpoint file.
             
        Raises:
            ValueError: If the attribute :code:`optimizer` is not set
            
        Returns:
            None
        """
        if self.optimizer is None:
            raise ValueError("You need to set an optimizer first by calling `set_optimizer`")

        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                self.checkpoint_folder, self.name.lower().replace(" ", "-") + ".ckpt")
        else:
            checkpoint_path = os.path.join(
                checkpoint_path, self.name.lower().replace(" ", "-") + ".ckpt")

        # First build the graph, then restore it
        self._build_graph()
        self._graph_built = True

        # Restore variables from disk.
        self._saver = tf.train.Saver()
        self._saver.restore(self._sess, checkpoint_path)

    @abstractmethod
    def partial_fit(self, X, y):
        pass

    @abstractmethod
    def _build_graph(self):
        pass


class NeuralNetwork(BaseTF):
    """docstring for NeuralNetwork"""

    def __init__(self, layers=None, batch_size=None, num_epochs=None,
                 optimizer=None, learning_rate=1, gradient_max_norm=100,
                 loss_function=None, random_seed=42, save_path=None):
        super(NeuralNetwork, self).__init__()
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.gradient_max_norm = gradient_max_norm
        self.optimizer = optimizer["class"](learning_rate=learning_rate)
        self.loss_function = loss_function["class"]
        self.layers = layers
        self.train_op = None
        self.input_shape = None
        self.batches_per_epoch = None
        self._training = None
        self.random_seed = random_seed
        tf.set_random_seed(self.random_seed)
        self.save_path = save_path

    def fit(self, data):
        self.batches_per_epoch = int(data.num_train_samples() / self.batch_size)
        self.input_shape = data.input_shape()
        self.output_shape = data.output_shape()
        self._init_session()
        X_eval_batch, y_eval_batch = data.get_eval_batch(512)
        for epoch in range(self.num_epochs):
            for batch in range(self.batches_per_epoch):
                X_batch, y_batch = data.get_train_batch(self.batch_size)
                self.partial_fit(X_batch, y_batch)
            loss = self.score(X_eval_batch, y_eval_batch)
            print(loss)
        return self

    def partial_fit(self, X, y):
        X = check_array(X, force_all_finite=True, allow_nd=True)
        loss, _ = self._sess.run([self.loss, self.train_op],
                                 feed_dict={self._X: X, self._y: y, self._training: True})
        return loss

    def predict(self, X):
        X = check_array(X, force_all_finite=True, allow_nd=True)
        prediction = self._sess.run([self.prediction],
                                 feed_dict={self._X: X, self._training: False})
        return prediction

    def predict_proba(self, X):
        pass

    def score(self, X, y):
        X = check_array(X, force_all_finite=True, allow_nd=True)
        loss = self._sess.run([self.loss],
                                 feed_dict={self._X: X, self._y: y, self._training: False})
        return loss

    def _build_graph(self):
        self.global_step = tf.Variable(0, trainable=False)
        self._X = tf.placeholder(
            tf.float32, [None] + utils.make_list(self.input_shape), name="X")
        self._y = tf.placeholder(
            tf.float32, [None] + utils.make_list(self.output_shape), name="y")
        self._training = tf.placeholder_with_default(
            True, shape=[], name="training")
     
        self.prediction = tf.contrib.layers.flatten(
        layers.build_architecture(x=self._X, architecture=self.layers, scope="layers",
            training=self._training))

        self.loss = self.loss_function(self.prediction, self._y)

        gradients, variables = zip(
            *self.optimizer.compute_gradients(self.loss))
        if self.gradient_max_norm is not None:
            gradients, _ = tf.clip_by_global_norm(
                gradients, self.gradient_max_norm)

        self.train_op = self.optimizer.apply_gradients(
            zip(gradients, variables), global_step=self.global_step, name="train_op")

def mean_cross_entropy(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))