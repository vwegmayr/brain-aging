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
        optimizer = tf.train.AdamOptimizer()
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


class MnistLogisticRegression(LogisticRegression):
    """
    def __init__(self, input_fn_config, config, params):
        super(MnistLogisticRegression, self).__init__(
            input_fn_config,
            config,
            params
        )
    """

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
