import tensorflow as tf
import sys
import os
import numpy as np


from modules.models.base import BaseTF
from modules.models.utils import custom_print
from src.logging import MetricLogger
import src.test_retest.regularizer as regularizer
from src.data.mnist import read as mnist_read
from src.train_hooks import ConfusionMatrixHook, ICCHook, BatchDumpHook, \
    RobustnessComputationHook, HookFactory, SumatraLoggingHook


def test_retest_evaluation_spec(
        labels, loss, preds_test,
        preds_retest, probs_test, probs_retest, params,
        mode, evaluation_hooks):

    return test_retest_two_labels_evaluation_spec(
        test_labels=labels, retest_labels=labels, loss=loss,
        preds_test=preds_test, preds_retest=preds_retest,
        probs_retest=probs_retest, probs_test=probs_test,
        params=params, mode=mode, evaluation_hooks=evaluation_hooks
    )


def test_retest_two_labels_evaluation_spec(
        test_labels, retest_labels, loss, preds_test,
        preds_retest, probs_test, probs_retest, params,
        mode, evaluation_hooks):
    # Evaluation
    eval_acc_test = tf.metrics.accuracy(
        labels=test_labels,
        predictions=preds_test
    )
    eval_acc_retest = tf.metrics.accuracy(
        labels=retest_labels,
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

    # Compute JS-divergence between test and retest probs
    js_divergences = regularizer.batch_divergence(
        probs_test,
        probs_retest,
        params["n_classes"],
        regularizer.js_divergence
    )

    # mean JS-divergence
    eval_js_mean = tf.metrics.mean(
        values=js_divergences
    )

    # std JS-divergence
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


def mnist_input_fn(X, data_params, y=None, train=True,
                   input_fn_config={}):
    """
    MNIST logistic regression for test-retest data. Loads presampled
    images from .npy files.
    """
    if train:
        # load training data
        images, labels = mnist_read.load_mnist_training(
            data_params["data_path"]
        )
        images = images[:data_params["train_size"]]
        labels = images[:data_params["train_size"]]

    else:
        # load test data
        images, labels = mnist_read.load_mnist_test(
            data_params["data_path"]
        )
        images = images[:data_params["test_size"]]
        labels = images[:data_params["test_size"]]

    return tf.estimator.inputs.numpy_input_fn(
        x={
            "X_0": images,
            "file_name_0": np.array([[str(i)] for i in range(len(images))])
        },
        y=labels,
        **input_fn_config,
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


def mnist_test_retest_two_labels_input_fn(X, data_params, y=None, train=True,
                                          input_fn_config={}):
    """
    MNIST logistic regression for test-retest data. Loads presampled
    images from .npy files.
    """
    if train:
        # load training data
        test, retest, test_labels, retest_labels = \
            mnist_read.load_test_retest_two_labels(
                data_params["data_path"],  # path to MNIST root folder
                data_params["train_test_retest"],
                data_params["train_size"],
                True,
                data_params["mix_pairs"]
            )

    else:
        # load test/retest data
        test, retest, test_labels, retest_labels = \
            mnist_read.load_test_retest_two_labels(
                data_params["data_path"],  # path to MNIST root folder
                data_params["test_test_retest"],
                data_params["test_size"],
                False,
                data_params["mix_pairs"]
            )

    labels = np.hstack((np.reshape(test_labels, (-1, 1)),
                        np.reshape(retest_labels, (-1, 1))))
    return tf.estimator.inputs.numpy_input_fn(
        x={
            "X_test": test,
            "X_retest": retest
        },
        y=labels,
        **input_fn_config
    )


def linear_trafo(X, out_dim, names, types=[tf.float32, tf.float32]):
    input_dim = X.get_shape()[1]
    w = tf.get_variable(
        name=names[0],
        shape=[input_dim, out_dim],
        dtype=types[0],
        initializer=tf.contrib.layers.xavier_initializer(seed=43)
    )

    b = tf.get_variable(
        name=names[1],
        shape=[1, out_dim],
        dtype=types[1],
        initializer=tf.initializers.zeros
    )

    Y = tf.add(
        tf.matmul(X, w),
        b,
        name=names[2]
    )

    return w, b, Y


def linear_trafo_multiple_input_tensors(Xs, out_dim, weight_names,
                                        output_names):
    """
    Performs a linear transformation with bias on all the input tensors.

    Args:
        - Xs: list of input tensors
        - out_dim: dimensionality of output
        - weight_names: names assigned to created weights
        - output_names: names assigned to output tensors

    Return:
        - w: created weighted matrix
        - b: created bias
        - Ys: list of output tensors
    """
    input_dim = Xs[0].get_shape()[1]
    w = tf.get_variable(
        name=weight_names[0],
        shape=[input_dim, out_dim],
        initializer=tf.contrib.layers.xavier_initializer(seed=43)
    )

    b = tf.get_variable(
        name=weight_names[1],
        shape=[1, out_dim],
        initializer=tf.initializers.zeros
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
        streamer=None,
        sumatra_params=None,
        hooks=None
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
        self.hooks = hooks

        # Initialize streamer
        self.streamer = None
        if streamer is not None:
            _class = streamer["class"]
            self.streamer = _class(**streamer["params"])

    def fit_main_training_loop(self, X, y):
        if self.streamer is not None:
            self.streamer.dump_split(self.save_path)
            self.streamer.dump_normalization(self.save_path)

        n_epochs = self.input_fn_config["num_epochs"]
        self.n_epochs = n_epochs
        self.input_fn_config["num_epochs"] = 1

        output_dir = self.config["model_dir"]
        self.metric_logger = MetricLogger(output_dir, "Evaluation metrics")

        for i in range(n_epochs):
            self.current_epoch = i
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

        self.streamer = None

    def get_hooks(self, preds_test, preds_retest, features_test,
                  features_retest):
        hooks = []

        if self.hooks is None:
            return hooks

        if ("icc_c1" in self.hooks) and (self.hooks["icc_c1"]):
            hook = ICCHook(
                icc_op=regularizer.per_feature_batch_ICC(
                    features_test,
                    features_retest,
                    regularizer.ICC_C1
                ),
                out_dir=os.path.join(self.save_path, "icc"),
                icc_name="ICC_C1"
            )
            hooks.append(hook)

        if ("confusion_matrix" in self.hooks) and \
                (self.hooks["confusion_matrix"]):
            hook = ConfusionMatrixHook(
                preds_test,
                preds_retest,
                self.params["n_classes"],
                os.path.join(self.save_path, "confusion")
            )
            hooks.append(hook)

        return hooks

    def get_batch_dump_hook(self, tensor_val, tensor_name):
        train_hook = BatchDumpHook(
            tensor_batch=tensor_val,
            batch_names=tensor_name,
            model_save_path=self.save_path,
            out_dir=self.data_params["dump_out_dir"],
            epoch=self.current_epoch,
            train=True
        )
        test_hook = BatchDumpHook(
            tensor_batch=tensor_val,
            batch_names=tensor_name,
            model_save_path=self.save_path,
            out_dir=self.data_params["dump_out_dir"],
            epoch=self.current_epoch,
            train=False
        )
        return train_hook, test_hook

    def get_robusntess_analysis_hook(self, feature_folder, train):
        hook = RobustnessComputationHook(
            model_save_path=self.save_path,
            out_dir=self.data_params["dump_out_dir"],
            epoch=self.current_epoch,
            train=train,
            feature_folder=feature_folder,
            robustness_streamer_config=self.hooks["robustness_streamer_config"]
        )

        return hook

    def get_mri_ae_hooks(self, reg_loss, rec_loss, enc_0, enc_1, features,
                         params):
        # Set up hooks
        train_hooks = []
        eval_hooks = []

        train_hook_names = params["train_hooks"]
        eval_hook_names = params["eval_hooks"]

        hook_factory = HookFactory(
            streamer=self.streamer,
            logger=self.metric_logger,
            out_dir=self.data_params["dump_out_dir"],
            model_save_path=self.save_path,
            epoch=self.current_epoch
        )

        if "embeddings" in train_hook_names:
            hidden_0_hook_train, hidden_0_hook_test = \
                hook_factory.get_batch_dump_hook(
                    enc_0, features["file_name_0"]
                )
            train_hooks.append(hidden_0_hook_train)
            eval_hooks.append(hidden_0_hook_test)

            hidden_1_hook_train, hidden_1_hook_test = \
                hook_factory.get_batch_dump_hook(
                    enc_1, features["file_name_1"]
                )
            train_hooks.append(hidden_1_hook_train)
            eval_hooks.append(hidden_1_hook_test)

            train_feature_folder = \
                hidden_0_hook_train.get_feature_folder_path()
            test_feature_folder = \
                hidden_0_hook_test.get_feature_folder_path()

        if "robustness" in train_hook_names:
            robustness_hook_train = self.get_robusntess_analysis_hook(
                feature_folder=train_feature_folder,
                train=True
            )
            robustness_hook_test = self.get_robusntess_analysis_hook(
                feature_folder=test_feature_folder,
                train=False
            )
            train_hooks.append(robustness_hook_train)
            eval_hooks.append(robustness_hook_test)

        if "predictions" in eval_hook_names:
            prediction_hook = hook_factory.get_prediction_hook(
                train_feature_folder=train_feature_folder,
                test_feature_folder=test_feature_folder,
                classify=True,
                target_label="healthy",
            )
            eval_hooks.append(prediction_hook)

            prediction_hook = hook_factory.get_prediction_hook(
                train_feature_folder=train_feature_folder,
                test_feature_folder=test_feature_folder,
                classify=False,
                target_label="age",
            )
            eval_hooks.append(prediction_hook)

            pred_robustness_hook = \
                hook_factory.get_prediction_robustness_hook()
            eval_hooks.append(pred_robustness_hook)

        # log embedding loss
        log_hook_train = SumatraLoggingHook(
            ops=[reg_loss, rec_loss],
            names=["hidden_reg_loss", "reconstruction_loss"],
            logger=self.metric_logger,
            namespace="train"
        )
        train_hooks.append(log_hook_train)

        log_hook_test = SumatraLoggingHook(
            ops=[reg_loss, rec_loss],
            names=["hidden_reg_loss", "reconstruction_loss"],
            logger=self.metric_logger,
            namespace="test"
        )
        eval_hooks.append(log_hook_test)

        if self.current_epoch == self.n_epochs - 1:
            eval_hooks.append(hook_factory.get_file_summarizer_hook(
                ["prediction_robustness", "predictions"]
            ))

        return train_hooks, eval_hooks

    def score(self, X, y):
        pass

    def export_estimator(self):
        pass
