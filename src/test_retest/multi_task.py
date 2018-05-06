import tensorflow as tf
import abc
import six


from src.test_retest.test_retest_base import \
    EvaluateEpochsBaseTF, \
    linear_trafo, \
    linear_trafo_multiple_input_tensors, \
    mnist_test_retest_input_fn
import src.test_retest.regularizer as regularizer


@six.add_metaclass(abc.ABCMeta)
class SoftTask(object):
    """
    Task of a multi-task model.
    """
    def __init__(self, weight):
        self.weight = weight
        self.prediction_op = None

    @abc.abstractmethod
    def loss(self, last_hidden_layer, targets, params):
        """
        Define output layer, loss and predictions. This method
        should also set the operation which outputs predictions.

        Args:
            - last_hidden_layer: last hidden layer of the shared model
            - targets: tensor containing targets
            - params: dictionary containing model parameters

        Return:
            - loss: loss to be minimized
        """
        pass

    def prediction(self):
        return self.prediction_op


class CrossEntropyTask(SoftTask):
    """
    Computes the cross-entropy loss for a given classification task.
    """
    def __init__(self, n_classes, *args, **kwargs):
        super(CrossEntropyTask, self).__init__(
            *args,
            **kwargs
        )
        self.n_classes = n_classes
        self.name = "CrossEntropy"

    def loss(self, last_hidden_layer, targets, params):
        with tf.name_scope("CrossEntropyTask"):
            _, _, logits = linear_trafo(
                last_hidden_layer,
                self.n_classes,
                ["weights", "bias", "logits"]
            )

            self.prediction_op = tf.argmax(
                input=logits,
                axis=1,
                name="predictions"
            )

            loss = tf.losses.sparse_softmax_cross_entropy(
                targets,
                logits
            )

            self.loss = loss

        return self.loss


class TestRetestCrossEntropyTask(SoftTask):
    """
    Computes the cross-entropy loss for a given test-retest
    classification task.
    """
    def __init__(self, n_classes, *args, **kwargs):
        super(TestRetestCrossEntropyTask, self).__init__(
            *args,
            **kwargs
        )
        self.n_classes = n_classes
        self.name = "TestRetestCrossEntropy"

    def loss(self, last_hidden_layer_test, last_hidden_layer_retest,
             targets, params):
        with tf.name_scope("TestRetestCrossEntropy"):
            w, b, logits = linear_trafo_multiple_input_tensors(
                [last_hidden_layer_test, last_hidden_layer_retest],
                self.n_classes,
                ["weights", "bias", "logits"],
                ["logits_test", "logits_retest"]
            )
            logits_test, logits_retest = logits

            self.prediction_test = tf.argmax(
                input=logits_test,
                axis=1,
                name="predictions_test"
            )

            self.prediction_retest = tf.argmax(
                input=logits_retest,
                axis=1,
                name="predictions_retest"
            )

            self.predictions_op = {
                "preds_test": self.prediction_test,
                "preds_retest": self.prediction_retest
            }

            loss_test = tf.losses.sparse_softmax_cross_entropy(
                targets,
                logits_test
            )
            loss_retest = tf.losses.sparse_softmax_cross_entropy(
                targets,
                logits_retest
            )

            self.loss = loss_test + loss_retest

        return self.loss


class TestRetestProbabilityTask(SoftTask):
    """
    Computes some loss on predicted probabilities
    """
    def __init__(self, n_classes, divergence_func, *args, **kwargs):
        super(TestRetestProbabilityTask, self).__init__(
            *args,
            **kwargs
        )
        self.n_classes = n_classes
        self.name = "TestRetestProbabilityTask"
        self.divergence_func = divergence_func

    def loss(self, last_hidden_layer_test, last_hidden_layer_retest,
             targets, params):
        with tf.name_scope("ProbabilityRegularizer"):
            w, b, logits = linear_trafo_multiple_input_tensors(
                [last_hidden_layer_test, last_hidden_layer_retest],
                self.n_classes,
                ["weights", "bias", "logits"],
                ["logits_test", "logits_retest"]
            )
            logits_test, logits_retest = logits

            self.prediction_test = tf.argmax(
                input=logits_test,
                axis=1,
                name="predictions_test"
            )

            self.prediction_retest = tf.argmax(
                input=logits_retest,
                axis=1,
                name="predictions_retest"
            )

            self.predictions_op = {
                "preds_test": self.prediction_test,
                "preds_retest": self.prediction_retest
            }

            self.probs_test = tf.nn.softmax(logits_test, "probs_test")
            self.probs_retest = tf.nn.softmax(logits_retest, "probs_retest")

            self.loss = regularizer.batch_divergence(
                self.probs_test,
                self.probs_retest,
                self.n_classes,
                self.divergence_func
            )

            self.loss = tf.reduce_mean(self.loss)

        return self.loss


class TestRetestJSDivergenceTask(TestRetestProbabilityTask):
    def __init__(self, *args, **kwargs):
        super(TestRetestJSDivergenceTask, self).__init__(
            divergence_func=regularizer.js_divergence,
            *args,
            **kwargs
        )
        self.name = "TestRetestJSDivergenceTask"


@six.add_metaclass(abc.ABCMeta)
class MTLSoftSharing(EvaluateEpochsBaseTF):
    """
    Base class for Multi-Task Learning with soft parameter
    sharing. Parameters are shared except for the output
    layer which is task specific.
    """
    def __init__(self, tasks, *args, **kwargs):
        super(MTLSoftSharing, self).__init__(
            *args,
            **kwargs
        )
        self.tasks = self.define_tasks(tasks)

    def define_tasks(self, tasks):
        """
        Create all the tasks.
        """
        task_objects = []
        for t in tasks:
            _class = t["class"]
            obj = _class(**t["params"])
            task_objects.append(obj)
        return task_objects

    @abc.abstractmethod
    def shared_layers(self, features, params):
        """
        Define the layers that shared among the tasks.

        Args:
            - features: dictionary containg feature tensors
            - params: dictionary containing model parameters

        Return:
            - last shared layer
        """
        pass

    def model_fn(self, features, labels, mode, params):
        layers = self.shared_layers(features, params)
        last_shared_layer = layers[-1]
        losses = [t.loss(last_shared_layer, labels, params)
                  for t in self.tasks]

        # Prediction
        predictions = {}
        for t in self.tasks:
            predictions[t.name] = t.prediction()

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions
            )

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


class MLTSoftTestRetest(MTLSoftSharing):

    def model_fn(self, features, labels, mode, params):
        layers = self.shared_layers(features, params)
        last_shared_layer_test, last_shared_layer_retest = layers[-1]

        losses = [t.loss(last_shared_layer_test, last_shared_layer_retest,
                  labels, params) for t in self.tasks]

        # Prediction
        predictions = {}
        for t in self.tasks:
            dic = t.prediction()
            for k in dic:
                k = t.name + "_" + k

            predictions.update(predictions)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions
            )

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


class MnistMLTSoftTestRetest(MLTSoftTestRetest):
    def shared_layers(self, features, params):
        exit("Not implemented")
        pass

    def gen_input_fn(self, X, y=None, train=True, input_fn_config={}):
        return mnist_test_retest_input_fn(
            X=X,
            y=y,
            data_params=self.data_params,
            train=train,
            input_fn_config=input_fn_config
        )
