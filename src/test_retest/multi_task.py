import tensorflow as tf
import abc
import six


from src.test_retest.test_retest_base import EvaluateEpochsBaseTF


@six.add_metaclass(abc.ABCMeta)
class Task(object):
    @abc.abstractmethod
    def loss(self, last_hidden_layer, features, labels, params):
        """
        Define output layer and loss.
        """
        pass

    @abs.abstractmethod
    def prediction(self):
        pass


@six.add_metaclass(abc.ABCMeta)
class MTLSoftSharing(EvaluateEpochsBaseTF):
    """
    Base class for Multi-Task Learning with soft parameter
    sharing. Parameters are shared except for the output
    layer which is task specific.
    """
    def __init__(self, *args, **kwargs):
        super(MTLSoftSharing, self).__init__(
            args,
            kwargs
        )
        self.tasks = self.define_tasks()

    @abs.abstractmethod
    def define_tasks(self):
        pass

    @abc.abstractmethod
    def shared_layers(self, features, labels, mode, params):
        pass

    @abc.abstractmethod
    def task_losses(self, last_layer, features, labels, mode, params):
        """
        Construct and
        """
        pass

    def model_fn(self, features, labels, mode, params, config):
        layers = self.shared_layers(features, labels, mode, params, config)
        last_shared_layer = layers[-1]
        losses = [t.loss(last_shared_layer, features, labels, params)
                  for t in self.tasks]

        # Compute total loss
        loss = 0
        for l in losses:
            loss += l

        # Minimize loss
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

        # Evaluation
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss
        )
