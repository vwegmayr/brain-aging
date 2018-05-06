import tensorflow as tf
import sys


from modules.models.base import BaseTF
from modules.models.utils import custom_print
from src.logging import MetricLogger
import src.test_retest.regularizer as regularizer
from src.data.mnist import read as mnist_read


def test_retest_evaluation_spec(labels, loss, preds_test, preds_retest,
                                probs_test, probs_retest, params, mode,
                                evaluation_hooks):
    # Evaluation
    eval_acc_test = tf.metrics.accuracy(
        labels=labels,
        predictions=preds_test
    )
    eval_acc_retest = tf.metrics.accuracy(
        labels=labels,
        predictions=preds_retest
    )
    # Compute KL-divergence between test and retest probs
    kl_divergences = regularizer.batch_divergence(
        probs_test,
        probs_retest,
        params["n_classes"],
        regularizer.kl_divergence
    )

    # mean KL-divergence
    eval_kl_mean = tf.metrics.mean(
        values=kl_divergences
    )

    # std KL-divergence
    _, kl_std = tf.nn.moments(kl_divergences, axes=[0])
    eval_kl_std = tf.metrics.mean(
        values=kl_std
    )

    # Compute KL-divergence between test and retest probs
    js_divergences = regularizer.batch_divergence(
        probs_test,
        probs_retest,
        params["n_classes"],
        regularizer.js_divergence
    )

    # mean KL-divergence
    eval_js_mean = tf.metrics.mean(
        values=js_divergences
    )

    # std KL-divergence
    _, js_std = tf.nn.moments(js_divergences, axes=[0])
    eval_js_std = tf.metrics.mean(
        values=js_std
    )

    eval_metric_ops = {
        'accuracy_test': eval_acc_test,
        'accuracy_retest': eval_acc_retest,
        'kl_mean': eval_kl_mean,
        'kl_std': eval_kl_std,
        'js_mean': eval_js_mean,
        'js_std': eval_js_std
    }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops,
        evaluation_hooks=evaluation_hooks
    )


def mnist_test_retest_input_fn(X, data_params, y=None, train=True,
                               input_fn_config={}):
    """
    MNIST logistic regression for test-retest data. Loads presampled
    images from .npy files.
    """
    if train:
        # load training data
        test, retest, labels = mnist_read.load_test_retest(
            data_params["data_path"],  # path to MNIST root folder
            data_params["train_test_retest"],
            data_params["train_size"],
            True
        )

    else:
        # load test/retest data
        test, retest, labels = mnist_read.load_test_retest(
            data_params["data_path"],  # path to MNIST root folder
            data_params["test_test_retest"],
            data_params["test_size"],
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


def linear_trafo(self, X, out_dim, names):
    input_dim = X.get_shape()[1]
    w = tf.get_variable(
        name=names[0],
        shape=[input_dim, out_dim],
        initializer=tf.contrib.layers.xavier_initializer(seed=43)
    )

    b = tf.get_variable(
        name=names[1],
        shape=[1, self.n_classes],
        initializer=tf.contrib.layers.xavier_initializer(seed=43)
    )

    Y = tf.add(
        tf.matmul(X, w),
        b,
        name=names[2]
    )

    return w, b, Y


def linear_trafo_multiple_input_tensors(self, Xs, out_dim, weight_names,
                                        output_names):
    input_dim = Xs[0].get_shape()[1]
    w = tf.get_variable(
        name=weight_names[0],
        shape=[input_dim, out_dim],
        initializer=tf.contrib.layers.xavier_initializer(seed=43)
    )

    b = tf.get_variable(
        name=weight_names[1],
        shape=[1, self.n_classes],
        initializer=tf.contrib.layers.xavier_initializer(seed=43)
    )

    Ys = []
    for i, X in enumerate(Xs):
        Y = tf.add(
            tf.matmul(X, w),
            b,
            name=output_names[i]
        )
        Ys.append(Y)

    return w, b, Ys


class EvaluateEpochsBaseTF(BaseTF):
    """
    Base estimator which is evluated every epoch.
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
        super(EvaluateEpochsBaseTF, self).__init__(
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
        self.metric_logger = MetricLogger(output_dir, "Evaluation metrics")

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
            self.metric_logger.add_evaluations("test", evaluation)

            if (self.sumatra_params is not None) and \
               (self.sumatra_params["log_train"]):
                # evaluation on training set
                evaluation_fn = self.gen_input_fn(
                    X, y, True, self.input_fn_config
                )
                evaluation = self.estimator.evaluate(input_fn=evaluation_fn)
                print(evaluation)
                self.metric_logger.add_evaluations("train", evaluation)

            # persist evaluations to json file
            self.metric_logger.dump()
            sys.stdout.flush()

    def score(self, X, y):
        pass
