import tensorflow as tf
import abc
import six
import os


from src.test_retest.test_retest_base import \
    EvaluateEpochsBaseTF, \
    linear_trafo, \
    linear_trafo_multiple_input_tensors, \
    mnist_test_retest_input_fn
import src.test_retest.regularizer as regularizer
from src.train_hooks import ConfusionMatrixHook


@six.add_metaclass(abc.ABCMeta)
class SoftTask(object):
    """
    Task of a multi-task model.
    """
    def __init__(self, weight):
        self.weight = weight
        self.prediction_op = None

    @abc.abstractmethod
    def compute_loss(self, last_hidden_layer, targets, params):
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

    def compute_loss(self, last_hidden_layer, targets, params):
        with tf.variable_scope("CrossEntropyTask"):
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

        return loss


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
        self.hooks = None

    def compute_loss(self, last_hidden_layer_test, last_hidden_layer_retest,
                     targets, params):
        with tf.variable_scope("TestRetestCrossEntropy"):
            w, b, logits = linear_trafo_multiple_input_tensors(
                [last_hidden_layer_test, last_hidden_layer_retest],
                self.n_classes,
                ["weights", "bias", "logits"],
                ["logits_test", "logits_retest"]
            )
            logits_test, logits_retest = logits

            prediction_test = tf.argmax(
                input=logits_test,
                axis=1,
                name="predictions_test"
            )

            prediction_retest = tf.argmax(
                input=logits_retest,
                axis=1,
                name="predictions_retest"
            )

            self.prediction_op = {
                "preds_test": prediction_test,
                "preds_retest": prediction_retest
            }

            loss_test = tf.losses.sparse_softmax_cross_entropy(
                targets,
                logits_test
            )
            loss_retest = tf.losses.sparse_softmax_cross_entropy(
                targets,
                logits_retest
            )

            loss = loss_test + loss_retest

        return loss


class TestRetestProbabilityTask(SoftTask):
    """
    Computes some loss on predicted probabilities. Also provides the
    addition of the cross-entropy loss and thereby forcing to output
    'correct' probabilities. Otherwise this task may predict probabilities
    independent of the true labels.
    """
    def __init__(self, n_classes, divergence_func, cross_entropy=True,
                 cross_entropy_weight=0.2, *args, **kwargs):
        super(TestRetestProbabilityTask, self).__init__(
            *args,
            **kwargs
        )
        self.n_classes = n_classes
        self.name = "TestRetestProbabilityTask"
        self.divergence_func = divergence_func
        self.cross_entropy = cross_entropy
        self.cross_entropy_weight = cross_entropy_weight

    def compute_loss(self, last_hidden_layer_test, last_hidden_layer_retest,
                     targets, params):
        with tf.variable_scope(self.__class__.__name__):
            w, b, logits = linear_trafo_multiple_input_tensors(
                [last_hidden_layer_test, last_hidden_layer_retest],
                self.n_classes,
                ["weights", "bias", "logits"],
                ["logits_test", "logits_retest"]
            )
            logits_test, logits_retest = logits

            prediction_test = tf.argmax(
                input=logits_test,
                axis=1,
                name="predictions_test"
            )

            prediction_retest = tf.argmax(
                input=logits_retest,
                axis=1,
                name="predictions_retest"
            )

            self.prediction_op = {
                "preds_test": prediction_test,
                "preds_retest": prediction_retest
            }

            probs_test = tf.nn.softmax(
                logits_test,
                name="probs_test"
            )
            probs_retest = tf.nn.softmax(
                logits_retest,
                name="probs_retest"
            )

            loss = regularizer.batch_divergence(
                probs_test,
                probs_retest,
                self.n_classes,
                self.divergence_func
            )

            loss = tf.reduce_mean(loss)

            if self.cross_entropy:
                cross_test = tf.losses.sparse_softmax_cross_entropy(
                    targets,
                    logits_test
                )
                cross_retest = tf.losses.sparse_softmax_cross_entropy(
                    targets,
                    logits_retest
                )
                loss += self.cross_entropy_weight * (cross_test + cross_retest)

        return loss


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
        self.task_list = self.define_tasks(tasks)

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
        losses = [t.weight * t.loss(last_shared_layer, labels, params)
                  for t in self.tasks]

        # Prediction
        predictions = {}
        for t in self.task_list:
            dic = t.prediction()
            t.prediction_op = None  # Make object pickable
            for k in dic:
                predictions[t.name + "_" + k] = dic[k]

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions
            )

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
        # Accuracy metrics
        eval_metric_ops = {}
        for k in predictions:
            eval_metric_ops["acc_" + k] = tf.metrics.accuracy(
                labels,
                predictions[k]
            )

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops
        )


class MLTSoftTestRetest(MTLSoftSharing):

    def model_fn(self, features, labels, mode, params):
        layers = self.shared_layers(features, params)
        last_shared_layer_test, last_shared_layer_retest = layers[-1]

        losses = [t.weight * t.compute_loss(last_shared_layer_test,
                  last_shared_layer_retest, labels, params)
                  for t in self.task_list]

        # Prediction
        predictions = {}
        cross_entropy_predictions = []
        for t in self.task_list:

            dic = t.prediction()
            t.prediction_op = None  # Make object pickable
            for k in dic:
                predictions[t.name + "_" + k] = dic[k]
            if isinstance(t, TestRetestCrossEntropyTask):
                keys = list(dic.keys())
                assert len(keys) == 2
                if keys[0].endswith("retest"):
                    cross_entropy_predictions = [dic[keys[1]], dic[keys[0]]]
                else:
                    cross_entropy_predictions = [dic[keys[0]], dic[keys[1]]]

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
        # Accuracy metrics
        eval_metric_ops = {}
        for k in predictions:
            eval_metric_ops["acc_" + k] = tf.metrics.accuracy(
                labels,
                predictions[k]
            )
        # Confusion matrix hook
        eval_hooks = []
        if len(cross_entropy_predictions) > 0:
            confusion_hook = ConfusionMatrixHook(
                cross_entropy_predictions[0],
                cross_entropy_predictions[1],
                params["n_classes"],
                os.path.join(self.save_path, "confusion")
            )
            eval_hooks.append(confusion_hook)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops,
            evaluation_hooks=eval_hooks
        )


class MnistMLTSoftTestRetestNoBody(MLTSoftTestRetest):
    def shared_layers(self, features, params):
        input_layer_test = tf.reshape(
            features["X_test"],
            [-1, self.params["input_dim"]]
        )

        input_layer_retest = tf.reshape(
            features["X_retest"],
            [-1, self.params["input_dim"]]
        )

        return [(input_layer_test, input_layer_retest)]

    def gen_input_fn(self, X, y=None, train=True, input_fn_config={}):
        return mnist_test_retest_input_fn(
            X=X,
            y=y,
            data_params=self.data_params,
            train=train,
            input_fn_config=input_fn_config
        )


class MnistMLTSoftTestRetestLinearBody(MnistMLTSoftTestRetestNoBody):
    def shared_layers(self, features, params):
        input_layer_test = tf.reshape(
            features["X_test"],
            [-1, self.params["input_dim"]]
        )

        input_layer_retest = tf.reshape(
            features["X_retest"],
            [-1, self.params["input_dim"]]
        )

        w, b, hidden = linear_trafo_multiple_input_tensors(
            Xs=[input_layer_test, input_layer_retest],
            out_dim=params["hidden_dim"],
            weight_names=["weight", "bias"],
            output_names=["hidden_test", "hidden_retest"]
        )

        return [(input_layer_test, input_layer_retest), (hidden[0], hidden[1])]


class MnistMLTSoftTestRetestNonLinearBody(MnistMLTSoftTestRetestNoBody):
    def shared_layers(self, features, params):
        input_layer_test = tf.reshape(
            features["X_test"],
            [-1, self.params["input_dim"]]
        )

        input_layer_retest = tf.reshape(
            features["X_retest"],
            [-1, self.params["input_dim"]]
        )

        f_test = tf.layers.dense(
            input_layer_test,
            units=self.params["hidden_dim"],
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(seed=43),
            name="layer_0"
        )

        f_retest = tf.layers.dense(
            input_layer_retest,
            units=self.params["hidden_dim"],
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(seed=43),
            name="layer_0",
            reuse=True
        )

        return [(input_layer_test, input_layer_retest), (f_test, f_retest)]
