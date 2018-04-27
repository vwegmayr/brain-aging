import tensorflow as tf


from modules.models.base import BaseTF
from modules.models.utils import custom_print
from src.data.mnist import read as mnist_read
from src.logging import MetricLogger


class LogisticRegression(BaseTF):
    def __init__(self, input_fn_config, config, params, data_params):
        super(LogisticRegression, self).__init__(
            input_fn_config,
            config,
            params
        )
        self.data_params = data_params

    def fit_main_training_loop(self, X, y):
        n_epochs = self.input_fn_config["num_epochs"]
        self.input_fn_config["num_epochs"] = 1

        output_dir = self.config["model_dir"]
        metric_logger = MetricLogger(output_dir, "Evaluation metrics")

        for i in range(n_epochs):
            # train
            self.estimator.train(
                input_fn=self.gen_input_fn(X, y, True, self.input_fn_config)
            )

            # evaluate
            evaluation_fn = self.gen_input_fn(
                X, y, False, self.input_fn_config
            )
            if evaluation_fn is None:
                custom_print("No evaluation - skipping evaluation.")
                return
            evaluation = self.estimator.evaluate(input_fn=evaluation_fn)

            metric_logger.add_evaluations(evaluation)
            metric_logger.dump()

    def model_fn(self, features, labels, mode, params):
        # Prediction
        input_layer = tf.reshape(
            features["X"],
            [-1, self.params["input_dim"]]
        )

        logits = tf.layers.dense(
            inputs=input_layer,
            units=self.params["n_classes"],
        )

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
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=preds)
        }

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops
        )

    def score(self, X, y):
        pass

    def export_estimator(self):
        pass


class TestRetestLogisticRegression(LogisticRegression):

    def prediction_nodes(self, features_tensor, labels, params):
        # Prediction
        input_layer = tf.reshape(
            features_tensor,
            [-1, self.params["input_dim"]]
        )

        logits = tf.layers.dense(
            inputs=input_layer,
            units=self.params["n_classes"],
        )

        probs = tf.nn.softmax(logits)
        preds = tf.argmax(input=logits, axis=1)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(preds, labels), tf.float32),
        )

        return input_layer, logits, probs, preds, accuracy

    def logits_regularizer(self, logits_1, logits_2):
        # Note that l2_loss only computes HALF the squared L2-norm
        return tf.nn.l2_loss(logits_1 - logits_2, name="l2_logits_diff")

    def model_fn(self, features, labels, mode, params):
        # Prediction
        # Construct nodes to performa logistic regression on test and retest data
        input_test, logits_test, probs_test, preds_test, \
            acc_test = self.prediction_nodes(
                features["X_test"],
                labels,
                params
            )

        input_retest, logits_retest, probs_retest, preds_retest, \
            acc_retest = self.prediction_nodes(
                features["X_retest"],
                labels,
                params
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
        regularizer = self.logits_regularizer(logits_test, logits_retest)

        # Training
        loss = loss_test + loss_retest + params["lambda"] * regularizer
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

    def gen_input_fn(self, X, y=None, train=True, input_fn_config={}):
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


class MnistTestRetestLogisticRegression(TestRetestLogisticRegression):
    def __init__(self, sampling, *args, **kwargs):
        self.sampling = sampling

        super(MnistTestRetestLogisticRegression, self).__init__(
            *args,
            **kwargs
        )

    def gen_input_fn(self, X, y=None, train=True, input_fn_config={}):
        if train:
            # load training data
            test, retest, labels = mnist_read.sample_test_retest_training(
                self.data_params["data_path"],
                self.sampling["train_size"],
                self.sampling["np_random_seed"]
            )

        else:
            # load test/retest data
            test, retest, labels = mnist_read.sample_test_retest_training(
                self.data_params["data_path"],
                self.sampling["test_size"],
                self.sampling["np_random_seed"]
            )

        return tf.estimator.inputs.numpy_input_fn(
            x={
                "X_test": test,
                "X_retest": retest
            },
            y=labels,
            **input_fn_config
        )
