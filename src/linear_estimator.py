import tensorflow as tf
import sys
import os


from modules.models.base import BaseTF
from modules.models.utils import custom_print
from src.data.mnist import read as mnist_read
from src.logging import MetricLogger
import src.regularizer as regularizer
from src.train_hooks import ConfusionMatrixHook, ICCHook


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

        bias = tf.get_variable(
            name="logistic_bias",
            shape=[1, self.params["n_classes"]],
            initializer=tf.contrib.layers.xavier_initializer(seed=43)
        )

        logits = tf.add(tf.matmul(input_layer, weights), bias, name="logits")

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
            reg = regularizer.l2(weights, "l2_weights")
        elif params["regularizer"] == "l2_sq":
            reg = regularizer.l2(weights, "l2_sq_weights")
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
        - l2_sq_logits: squared of l2_logits
        - l2_probs: l2-norm of the difference between test
          and retest probabilities
        - l2_sq_probs: squared of l2_probs
        - l1_probs: l1-norm of the difference between test
          and retest probabilities
    """
    def prediction_nodes(self, features_tensor, labels, params, weights, bias):
        """
        Args:
            - features_tensor: tensor representing features
            - labels: tensorf representing labels
            - params: dictionary containg parameters for the model function
            - weights: weight tensor used to compute logits
            - bias: bias tensor

        Returns:
            - input_layer: tensor representing input layer
            - logits: output tensor of logits computation
            - probs: output tensor of predicted probabilities
            - preds: output tensor of predicted classes
        """
        input_layer = tf.reshape(
            features_tensor,
            [-1, self.params["input_dim"]]
        )

        logits = tf.add(tf.matmul(input_layer, weights), bias, name="logits")

        probs = tf.nn.softmax(logits, name="probabilities")
        preds = tf.argmax(input=logits, axis=1, name="predictions")
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(preds, labels), tf.float32),
            name="accuracy"
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

        bias = tf.get_variable(
            name="logistic_bias",
            shape=[1, self.params["n_classes"]],
            initializer=tf.contrib.layers.xavier_initializer(seed=43)
        )

        with tf.name_scope("test"):
            input_test, logits_test, probs_test, preds_test, \
                acc_test = self.prediction_nodes(
                    features["X_test"],
                    labels,
                    params,
                    weights,
                    bias
                )

        with tf.name_scope("retest"):
            input_retest, logits_retest, probs_retest, preds_retest, \
                acc_retest = self.prediction_nodes(
                    features["X_retest"],
                    labels,
                    params,
                    weights,
                    bias
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
            reg_w = regularizer.l2(weights, "l2_weights")
        elif params["weight_regularizer"] == "l2_sq":
            reg_w = regularizer.l2_squared(weights, "l2_sq_weights")
        elif params["weight_regularizer"] == "l1":
            reg_w = regularizer.l1(weights, "l1_weights")
        else:
            raise ValueError("Regularizer not found")

        reg_w *= params["lambda_w"]

        # output
        key = "output_regularizer"
        if params[key] is None:
            reg_out = 0
        elif params[key] == "l2_logits":
            reg_out = regularizer.l2(
                logits_test - logits_retest,
                name="l2_logits_diff"
            )
        elif params[key] == "l2_sq_logits":
            reg_out = regularizer.l2_squared(
                logits_test - logits_retest,
                name="l2_sq_logits_diff"
            )
        elif params[key] == "l2_probs":
            reg_out = regularizer.l2(
                probs_test - probs_retest,
                name="l2_probs_diff"
            )
        elif params[key] == "l2_sq_probs":
            reg_out = regularizer.l2_squared(
                probs_test - probs_retest,
                name="l2_sq_probs_diff"
            )
        elif params[key] == "l1_probs":
            reg_out = regularizer.l1(
                probs_test - probs_retest,
                name="l1_probs_diff"
            )
        elif params[key] == "kl_divergence":
            reg_out = regularizer.batch_divergence(
                probs_test,
                probs_retest,
                params["n_classes"],
                regularizer.kl_divergence
            )
            reg_out = tf.reduce_mean(reg_out)
        elif params[key] == "js_divergence":
            reg_out = regularizer.batch_divergence(
                probs_test,
                probs_retest,
                params["n_classes"],
                regularizer.kl_divergence
            )
            reg_out = tf.reduce_mean(reg_out)
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

        confusion_hook = ConfusionMatrixHook(
            preds_test,
            preds_retest,
            params["n_classes"],
            os.path.join(self.save_path, "confusion")
        )
        # Evaluation
        return test_retest_evaluation_spec(
            labels=labels,
            loss=loss,
            preds_test=preds_test,
            preds_retest=preds_retest,
            probs_test=probs_test,
            probs_retest=probs_retest,
            params=params,
            mode=mode,
            evaluation_hooks=[confusion_hook]
        )


class TestRetestTwoLevelLogisticRegression(LogisticRegression):
    """
    TODO: make first transformation configurable
    Use one hidden layer to learn an intermediate representation
    via a (linear) transformation. This hidden representation
    can be interpreted as new features which can be regularized
    on test-retest pairs.
    """
    def prediction_nodes(self, features_tensor, labels, params, w_1, w_2,
                         b_1, b_2):
        """
        Args:
            - features_tensor: tensor representing features
            - labels: tensorf representing labels
            - params: dictionary containg parameters for the model function
            - w1: weight tensor used for first-level transformation
            - w2: weight tensor used for logistic regression
            - b1: first-level bias tensor
            - b2: logistic regression bias tensor

        Returns:
            - input_layer: tensor representing input layer
            - logits: output tensor of logits computation
            - probs: output tensor of predicted probabilities
            - preds: output tensor of predicted classes
            - hidden_features: hidden layer activations
        """
        input_layer = tf.reshape(
            features_tensor,
            [-1, self.params["input_dim"]]
        )

        hidden_features = tf.add(
            tf.matmul(input_layer, w_1),
            b_1,
            name="hidden_features"
        )
        logits = tf.add(
            tf.matmul(hidden_features, w_2),
            b_2,
            name="logits"
        )

        probs = tf.nn.softmax(logits, name="probabilities")
        preds = tf.argmax(input=logits, axis=1, name="predictions")
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(preds, labels), tf.float32),
            name="accuracy"
        )

        return input_layer, hidden_features, logits, probs, preds, accuracy

    def model_fn(self, features, labels, mode, params):
        print(features["X_test"])
        exit()
        # Construct first-layer weights
        hidden_weights = tf.get_variable(
            name="hidden_weights",
            shape=[self.params["input_dim"], self.params["hidden_dim"]],
            initializer=tf.contrib.layers.xavier_initializer(seed=43)
        )

        hidden_bias = tf.get_variable(
            name="hidden_bias",
            shape=[1, self.params["hidden_dim"]],
            initializer=tf.contrib.layers.xavier_initializer(seed=43)
        )        

        # Construct second-layer weights
        logistic_weights = tf.get_variable(
            name="logistic_weights",
            shape=[self.params["hidden_dim"], self.params["n_classes"]],
            initializer=tf.contrib.layers.xavier_initializer(seed=43)
        )

        logistic_bias = tf.get_variable(
            name="logistic_bias",
            shape=[1, self.params["n_classes"]],
            initializer=tf.contrib.layers.xavier_initializer(seed=43)
        )

        with tf.name_scope("test"):
            input_test, hidden_features_test, logits_test, probs_test, \
                preds_test, acc_test = self.prediction_nodes(
                    features["X_test"],
                    labels,
                    params,
                    hidden_weights,
                    logistic_weights,
                    hidden_bias,
                    logistic_bias
                )

        with tf.name_scope("retest"):
            input_retest, hidden_features_retest, logits_retest, probs_retest,\
                preds_retest, acc_retest = self.prediction_nodes(
                    features["X_retest"],
                    labels,
                    params,
                    hidden_weights,
                    logistic_weights,
                    hidden_bias,
                    logistic_bias
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

        # hidden features regularization
        f1 = hidden_features_test
        f2 = hidden_features_retest

        param_name = "hidden_features_regularizer"
        if params[param_name] is None:
            reg_f = 0
        elif params[param_name] == "l2":
            reg_f = regularizer.l2(
                f1 - f2,
                name="l2_hidden_features"
            )
        elif params[param_name] == "l2_sq":
            reg_f = regularizer.l2_squared(
                f1 - f2,
                name="l2_sq_hidden_features"
            )
        else:
            raise ValueError("Regularizer not found")

        reg_f *= params["lambda_f"]

        # Training
        loss = loss_test + loss_retest + reg_f
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
        confusion_hook = ConfusionMatrixHook(
            preds_test,
            preds_retest,
            params["n_classes"],
            os.path.join(self.save_path, "confusion")
        )

        icc_hook = ICCHook(
            icc_op=regularizer.per_feature_batch_ICC(
                hidden_features_test,
                hidden_features_retest,
                regularizer.ICC_C1
            ),
            out_dir=os.path.join(self.save_path, "icc"),
            icc_name="ICC_C1"
        )

        return test_retest_evaluation_spec(
            labels=labels,
            loss=loss,
            preds_test=preds_test,
            preds_retest=preds_retest,
            probs_test=probs_test,
            probs_retest=probs_retest,
            params=params,
            mode=mode,
            evaluation_hooks=[confusion_hook, icc_hook]
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


class MnistTestRetestTwoLevelLogisticRegression(
        TestRetestTwoLevelLogisticRegression):
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
