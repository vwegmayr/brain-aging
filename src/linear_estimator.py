import tensorflow as tf
import sys


from modules.models.base import BaseTF
from modules.models.utils import custom_print
from src.data.mnist import read as mnist_read
from src.logging import MetricLogger


class LogisticRegression(BaseTF):
    """
    Base estimator to preform logistic regression.
    Allow the following regularization:
        - l2: l2-norm of weights
    """
    def __init__(
        self,
        input_fn_config,
        config,
        params,
        data_params,
        sumatra_params=None
    ):
        """
        Args:
            - input_fn_config: configuration for tf input function of
              tf estimator
            - config: configuration for tf estimator constructor
            - params: parameters for the model functions of the tf
              estimator
            - data_params: should contain information about the
              the data location that should be read
            - sumatra_params: contains information about sumatra logging
        """
        super(LogisticRegression, self).__init__(
            input_fn_config,
            config,
            params
        )
        self.data_params = data_params
        self.sumatra_params = sumatra_params

    def fit_main_training_loop(self, X, y):
        n_epochs = self.input_fn_config["num_epochs"]
        self.input_fn_config["num_epochs"] = 1

        output_dir = self.config["model_dir"]
        metric_logger = MetricLogger(output_dir, "Evaluation metrics")
        self.params["logger"] = metric_logger

        for i in range(n_epochs):
            # train
            self.estimator.train(
                input_fn=self.gen_input_fn(X, y, True, self.input_fn_config)
            )

            # evaluate
            # evaluation on test set
            evaluation_fn = self.gen_input_fn(
                X, y, False, self.input_fn_config
            )
            if evaluation_fn is None:
                custom_print("No evaluation - skipping evaluation.")
                return
            evaluation = self.estimator.evaluate(input_fn=evaluation_fn)
            print(evaluation)
            metric_logger.add_evaluations("test", evaluation)

            if (self.sumatra_params is not None) and \
               (self.sumatra_params["log_train"]):
                # evaluation on training set
                evaluation_fn = self.gen_input_fn(
                    X, y, True, self.input_fn_config
                )
                evaluation = self.estimator.evaluate(input_fn=evaluation_fn)
                print(evaluation)
                metric_logger.add_evaluations("train", evaluation)

            # persist evaluations to json file
            metric_logger.dump()
            sys.stdout.flush()

    def model_fn(self, features, labels, mode, params):
        # Prediction
        input_layer = tf.reshape(
            features["X"],
            [-1, self.params["input_dim"]]
        )

        weights = tf.get_variable(
            name="logistic_weights",
            shape=[self.params["input_dim"], self.params["n_classes"]],
            initializer=tf.contrib.layers.xavier_initializer(seed=43)
        )

        logits = tf.matmul(input_layer, weights)

        probs = tf.nn.softmax(logits)
        preds = tf.argmax(input=logits, axis=1)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(preds, labels), tf.float32),
            name="acc_tensor"
        )

        predictions = {
            "classes": preds,
            "probs": probs,
            "accuracy": accuracy
        }

        metric_ops = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=preds)
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions
            )

        # Training
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels,
            logits=logits
        )

        if params["regularizer"] is None:
            reg = 0
        elif params["regularizer"] == "l2":
            reg = tf.norm(weights, ord=2)
        else:
            raise ValueError("Regularizer not found")

        lam = params["lambda"]
        loss += lam * reg

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
            loss=loss,
            eval_metric_ops=metric_ops
        )

    def score(self, X, y):
        pass

    def export_estimator(self):
        pass


class TestRetestLogisticRegression(LogisticRegression):
    """
    Logistic regression for test retest data.
    Allow the following regularization:
        - l2_logits: l2-norm of the difference between test
          and retest logits
    """
    def prediction_nodes(self, features_tensor, labels, params, weights):
        # Prediction
        input_layer = tf.reshape(
            features_tensor,
            [-1, self.params["input_dim"]]
        )

        logits = tf.matmul(input_layer, weights)

        probs = tf.nn.softmax(logits)
        preds = tf.argmax(input=logits, axis=1)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(preds, labels), tf.float32),
        )

        return input_layer, logits, probs, preds, accuracy

    def model_fn(self, features, labels, mode, params):
        # Prediction
        # Construct nodes to performa logistic regression on test and retest data
        weights = tf.get_variable(
            name="logistic_weights",
            shape=[self.params["input_dim"], self.params["n_classes"]],
            initializer=tf.contrib.layers.xavier_initializer(seed=43)
        )

        input_test, logits_test, probs_test, preds_test, \
            acc_test = self.prediction_nodes(
                features["X_test"],
                labels,
                params,
                weights
            )

        input_retest, logits_retest, probs_retest, preds_retest, \
            acc_retest = self.prediction_nodes(
                features["X_retest"],
                labels,
                params,
                weights
            )

        predictions = {
            "classes_test": preds_test,
            "classes_retest": preds_retest
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions
            )

        # compute cross-entropy loss for test and retest
        loss_test = tf.losses.sparse_softmax_cross_entropy(
            labels,
            logits=logits_test
        )

        loss_retest = tf.losses.sparse_softmax_cross_entropy(
            labels=labels,
            logits=logits_retest
        )

        # regularization
        # weights
        if params["weight_regularizer"] is None:
            reg_w = 0
        elif params["weight_regularizer"] == "l2":
            reg_w = tf.norm(weights, ord=2)
        elif params["weight_regularizer"] == "l1":
            reg_w = tf.norm(weights, ord=1)
        else:
            raise ValueError("Regularizer not found")

        reg_w *= params["lambda_w"]

        # output
        if params["output_regularizer"] is None:
            reg_out = 0
        elif params["output_regularizer"] == "l2_logits":
            reg_out = tf.norm(
                logits_test - logits_retest,
                ord=2,
                name="l2_logits_diff"
            )
        elif params["output_regularizer"] == "l2_probs":
            reg_out = tf.norm(
                probs_test - probs_retest,
                ord=2,
                name="l2_probs_diff"
            )
        elif params["output_regularizer"] == "l1_probs":
            reg_out = tf.norm(
                probs_test - probs_retest,
                ord=1,
                name="l1_probs_diff"
            )
        else:
            raise ValueError("Regularizer not found")

        reg_out *= params["lambda_o"]

        # Training
        loss = loss_test + loss_retest + reg_w + reg_out
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
        eval_acc_test = tf.metrics.accuracy(
            labels=labels,
            predictions=preds_test
        )
        eval_acc_retest = tf.metrics.accuracy(
            labels=labels,
            predictions=preds_retest
        )
        eval_metric_ops = {
            'accuracy_test': eval_acc_test,
            'accuracy_retest': eval_acc_retest
        }

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops
        )


class MnistLogisticRegression(LogisticRegression):
    """
    Logistic regression for the MNIST dataset.
    """
    def gen_input_fn(self, X, y=None, train=True, input_fn_config={}):
        """
        Loads train and test images from the MNIST root data folder
        containing the raw MNIST files.
        """
        if train:
            # load training data
            images, labels = mnist_read.load_mnist_training(
                self.data_params["data_path"]
            )

        else:
            # load test data
            images, labels = mnist_read.load_mnist_test(
                self.data_params["data_path"]
            )

        return tf.estimator.inputs.numpy_input_fn(
            x={"X": images},
            y=labels,
            **input_fn_config
        )

    def input_fn(self, X, y, batch_size=64, train=True):
        if train:
            # load training data
            images, labels = mnist_read.load_mnist_training(
                self.data_params["data_path"]
            )

            dataset = tf.data.Dataset.from_tensor_slices((images, labels))
            dataset.shuffle(1000).batch(batch_size)

        return dataset


class MnistTestRetestLogisticRegression(TestRetestLogisticRegression):
    """
    MNIST logistic regression for test-retest data. Loads presampled
    images from .npy files.
    """
    def gen_input_fn(self, X, y=None, train=True, input_fn_config={}):
        if train:
            # load training data
            test, retest, labels = mnist_read.load_test_retest(
                self.data_params["data_path"],  # path to MNIST root folder
                self.data_params["train_test_retest"],
                self.data_params["train_size"],
                True
            )

        else:
            # load test/retest data
            test, retest, labels = mnist_read.load_test_retest(
                self.data_params["data_path"],  # path to MNIST root folder
                self.data_params["test_test_retest"],
                self.data_params["test_size"],
                False
            )

        return tf.estimator.inputs.numpy_input_fn(
            x={
                "X_test": test,
                "X_retest": retest
            },
            y=labels,
            **input_fn_config
        )
