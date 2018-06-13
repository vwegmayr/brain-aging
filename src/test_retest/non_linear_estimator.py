import tensorflow as tf


from src.test_retest.test_retest_base import EvaluateEpochsBaseTF
import src.test_retest.regularizer as regularizer
from src.test_retest.test_retest_base import \
    test_retest_two_labels_evaluation_spec
from src.test_retest.test_retest_base \
    import mnist_test_retest_two_labels_input_fn


def name_to_hidden_regularization(layer_id, reg_name, activations_test,
                                  activations_retest):
    if reg_name == regularizer.JS_DIVERGENCE_LABEL:
        s_test = tf.nn.softmax(
            activations_test,
            name=str(layer_id) + "_softmax_test"
        )
        s_retest = tf.nn.softmax(
            activations_retest,
            name=str(layer_id) + "_softmax_retest"
        )
        n = activations_test.get_shape().as_list()[1]
        batch_div = regularizer.batch_divergence(
            s_test,
            s_retest,
            n,
            regularizer.js_divergence
        )
        return tf.reduce_mean(batch_div)

    elif reg_name == regularizer.L2_SQUARED_LABEL:
        return regularizer.l2_squared_mean_batch(
                    activations_test - activations_retest,
                    name=str(layer_id) + "_l2_activations"
               )
    elif reg_name == regularizer.COSINE_SIMILARITY:
        similarities = regularizer.cosine_similarities(
            activations_test,
            activations_retest
        )
        return tf.reduce_mean(similarities)
    else:
        raise ValueError("regularization name '{}' is unknown".format(
                         reg_name))


class DeepTestRetestClassifier(EvaluateEpochsBaseTF):
    def model_fn(self, features, labels, mode, params):
        input_test = tf.reshape(
            features["X_test"],
            [-1, self.params["input_dim"]]
        )
        input_retest = tf.reshape(
            features["X_retest"],
            [-1, self.params["input_dim"]]
        )

        test_labels = labels[:, 0]
        retest_labels = labels[:, 1]

        f_test = input_test
        f_retest = input_retest
        hidden_sizes = params["hidden_layer_sizes"]

        if "hidden_layer_regularizers" in params:
            hidden_regs = params["hidden_layer_regularizers"]
        else:
            hidden_regs = None
        if "hidden_lambdas" in params:
            hidden_lambdas = params["hidden_lambdas"]
        else:
            hidden_lambdas = None

        predictions = {}

        loss_f = 0
        # Construct hidden layers
        for i, s in enumerate(hidden_sizes):
            name = "layer_" + str(i)
            f_test = tf.layers.dense(
                f_test,
                units=s,
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(
                    seed=43
                ),
                name=name
            )

            f_retest = tf.layers.dense(
                f_retest,
                units=s,
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(
                    seed=43
                ),
                name=name,
                reuse=True
            )

            # Optional Regularization
            if (hidden_lambdas is not None) and (hidden_lambdas[i] != 0):
                print("regularization on layer {}".format(i))
                hidden_loss = name_to_hidden_regularization(
                    layer_id=i,
                    reg_name=hidden_regs[i],
                    activations_test=f_test,
                    activations_retest=f_retest
                )
                loss_f += hidden_lambdas[i] * hidden_loss

            # Allows computation of activations when loading a trained model
            # and executing it in prediction mode
            predictions.update({
                "hidden_features_test_" + name: f_test,
                "hidden_features_retest_" + name: f_retest
            })

        # Compute logits and predictions
        n_classes = params["n_classes"]
        logits_test = tf.layers.dense(
            f_test,
            units=n_classes,
            kernel_initializer=tf.contrib.layers.xavier_initializer(seed=43),
            name="logits"
        )
        logits_retest = tf.layers.dense(
            f_retest,
            units=n_classes,
            kernel_initializer=tf.contrib.layers.xavier_initializer(seed=43),
            name="logits",
            reuse=True
        )

        preds_test = tf.argmax(logits_test, axis=1, name="pred_test")
        preds_retest = tf.argmax(logits_retest, axis=1, name="pred_retest")

        predictions.update({
            "classes_test": preds_test,
            "classes_retest": preds_retest,
        })

        probs_test = tf.nn.softmax(logits_test, name="probs_test")
        probs_retest = tf.nn.softmax(logits_retest, name="probs_retest")

        predictions.update({
            "probs_test": probs_test,
            "probs_retest": probs_retest 
        })

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions
            )

        # compute cross-entropy loss for test and retest
        loss_test = tf.losses.sparse_softmax_cross_entropy(
            labels=test_labels,
            logits=logits_test
        )

        loss_retest = tf.losses.sparse_softmax_cross_entropy(
            labels=retest_labels,
            logits=logits_retest
        )

        # output regularization
        key = "output_regularizer"
        if params[key] is None:
            loss_o = 0
        elif params[key] == "kl_divergence":
            loss_o = regularizer.batch_divergence(
                probs_test,
                probs_retest,
                params["n_classes"],
                regularizer.kl_divergence
            )
            loss_o = tf.reduce_mean(loss_o)
        elif params[key] == "js_divergence":
            loss_o = regularizer.batch_divergence(
                probs_test,
                probs_retest,
                params["n_classes"],
                regularizer.kl_divergence
            )
            loss_o = tf.reduce_mean(loss_o)
        else:
            raise ValueError("Regularizer not found")

        loss_o = params["lambda_o"] * loss_o

        # Sum up different regularizations
        loss = loss_test + loss_retest + loss_o + loss_f

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
        eval_hooks = self.get_hooks(
            preds_test=preds_test,
            preds_retest=preds_retest,
            features_test=f_test,
            features_retest=f_retest
        )

        return test_retest_two_labels_evaluation_spec(
            test_labels=test_labels,
            retest_labels=retest_labels,
            loss=loss,
            preds_test=preds_test,
            preds_retest=preds_retest,
            probs_test=probs_test,
            probs_retest=probs_retest,
            params=params,
            mode=mode,
            evaluation_hooks=eval_hooks
        )


class MnistDeepTestRetestClassifier(DeepTestRetestClassifier):
    """
    MNIST logistic regression for test-retest data. Loads presampled
    images from .npy files.
    """
    def gen_input_fn(self, X, y=None, train=True, input_fn_config={}):
        return mnist_test_retest_two_labels_input_fn(
            X=X,
            y=y,
            data_params=self.data_params,
            train=train,
            input_fn_config=input_fn_config
        )
