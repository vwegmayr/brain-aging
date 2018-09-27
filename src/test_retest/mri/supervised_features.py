import tensorflow as tf
import abc
import pydoc

from src.test_retest.test_retest_base import EvaluateEpochsBaseTF
from .model_components import MultiLayerPairEncoder, \
    PairClassificationHead, Conv3DEncoder
from src.train_hooks import SumatraLoggingHook, HookFactory


class PairClassification(EvaluateEpochsBaseTF):
    @abc.abstractmethod
    def get_encodings(self, features, params, mode):
        pass

    def model_fn(self, features, labels, mode, params):
        enc_0, enc_1 = self.get_encodings(features, params, mode)

        clf = PairClassificationHead(
            features=features,
            params=params,
            encodings=[enc_0, enc_1]
        )

        preds_0, preds_1 = clf.get_predictions()
        labels_0, labels_1 = clf.get_labels()
        all_preds = tf.concat([preds_0, preds_1], axis=0)
        all_labels = tf.concat([labels_0, labels_1], axis=0)

        # Make predictions
        predictions = {
            "encoding": enc_0,
            "classes": preds_0
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions
            )

        loss = clf.get_total_loss()

        optimizer = tf.train.AdamOptimizer(
            learning_rate=params["learning_rate"]
        )
        train_op = optimizer.minimize(loss, tf.train.get_global_step())

        # Hooks
        train_hooks = []
        eval_hooks = []

        if params["validation"]:
            eval_name = "validation"
            validation = True
        else:
            eval_name = "test"
            validation = False

        train_hook_names = params["train_hooks"]
        eval_hook_names = params["eval_hooks"]

        hf = HookFactory(
            streamer=self.streamer,
            logger=self.metric_logger,
            out_dir=self.data_params["dump_out_dir"],
            model_save_path=self.save_path,
            epoch=self.current_epoch
        )

        if "embeddings" in train_hook_names and not validation:
            enc_0_hook_train, enc_0_hook_test = \
                hf.get_batch_dump_hook(enc_0, features["file_name_0"])
            train_hooks.append(enc_0_hook_train)
            eval_hooks.append(enc_0_hook_test)

            enc_1_hook_train, enc_1_hook_test = \
                hf.get_batch_dump_hook(enc_1, features["file_name_1"])
            train_hooks.append(enc_1_hook_train)
            eval_hooks.append(enc_1_hook_test)

            train_feature_folder = enc_0_hook_train.get_feature_folder_path()
            test_feature_folder = enc_0_hook_test.get_feature_folder_path()

        if "robustness" in train_hook_names and not validation:
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

            # Robustness of head predictions
            hook = hf.get_tensor_prediction_robustness_hook(
                [preds_0, preds_1],
                [features["file_name_0"], features["file_name_1"]],
                "head_prediction",
                params["target_label_key"],
                True
            )
            train_hooks.append(hook)

            hook = hf.get_tensor_prediction_robustness_hook(
                [preds_0, preds_1],
                [features["file_name_0"], features["file_name_1"]],
                "head_prediction",
                params["target_label_key"],
                False
            )
            eval_hooks.append(hook)

        if "predictions" in eval_hook_names and not validation:
            prediction_hook = hf.get_prediction_hook(
                train_feature_folder=train_feature_folder,
                test_feature_folder=test_feature_folder,
                classify=True,
                target_label="healthy",
            )
            eval_hooks.append(prediction_hook)

            prediction_hook = hf.get_prediction_hook(
                train_feature_folder=train_feature_folder,
                test_feature_folder=test_feature_folder,
                classify=False,
                target_label="age",
            )
            eval_hooks.append(prediction_hook)

            pred_robustness_hook = \
                hf.get_prediction_robustness_hook()
            eval_hooks.append(pred_robustness_hook)

        acc = clf.get_accuracy()
        all_losses, loss_names = clf.get_losses_with_names()

        log_hook_train = SumatraLoggingHook(
            ops=[loss, acc] + all_losses,
            names=["loss", "acc"] + loss_names,
            logger=self.metric_logger,
            namespace="train"
        )
        train_hooks.append(log_hook_train)
        """
        log_hook_test = SumatraLoggingHook(
            ops=all_losses,
            names=["acc"] + loss_names,
            logger=self.metric_logger,
            namespace="test"
        )
        eval_hooks.append(log_hook_test)
        """

        if self.current_epoch == self.n_epochs - 1 and not validation:
            eval_hooks.append(hf.get_file_summarizer_hook(
                ["prediction_robustness", "predictions",
                 "head_prediction"]
            ))

            if "robustness" in eval_hook_names:
                enc_dim = enc_0.get_shape().as_list()[-1]
                diag_dim = params["diagnose_dim"]
                not_reg = set([str(i) for i in range(0, enc_dim - diag_dim)])
                reg = set([str(i) for i in range(enc_dim - diag_dim, enc_dim)])
                eval_hooks.append(hf.get_compare_regularized_unregularized_features(
                    reg=reg,
                    not_reg=not_reg
                ))

        eval_metric_ops = {
            "acc": tf.metrics.mean(acc),
            "recall": tf.metrics.recall(
                labels=all_labels,
                predictions=all_preds
            ),
            "specificity": tf.metrics.recall(
                labels=1 - all_labels,
                predictions=1 - all_preds
            )
        }
        for op, l_name in zip(all_losses, loss_names):
            eval_metric_ops[l_name] = tf.metrics.mean(op)

        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                training_hooks=train_hooks,
                eval_metric_ops=eval_metric_ops,
            )

        return tf.estimator.EstimatorSpec(
            mode=mode,
            eval_metric_ops=eval_metric_ops,
            loss=loss,
            evaluation_hooks=eval_hooks
        )

    def gen_input_fn(self, X, y=None, train=True, input_fn_config={}):
        return self.streamer.get_input_fn(train)


class LinearPairClassification(PairClassification):
    def get_encodings(self, features, params, mode):
        encoder = MultiLayerPairEncoder(
            features=features,
            params=params,
            streamer=self.streamer
        )

        enc_0, enc_1 = encoder.get_encodings()
        return enc_0, enc_1


class ConvPairClassification(PairClassification):
    def get_encodings(self, features, params, mode):
        with tf.variable_scope("conv_3d_encoder", reuse=tf.AUTO_REUSE):
            encoder_0 = Conv3DEncoder(
                input_key="X_0",
                features=features,
                params=params,
                streamer=self.streamer
            )

        with tf.variable_scope("conv_3d_encoder", reuse=tf.AUTO_REUSE):
            encoder_1 = Conv3DEncoder(
                input_key="X_1",
                features=features,
                params=params,
                streamer=self.streamer
            )

        z_0 = encoder_0.get_encoding()
        z_1 = encoder_1.get_encoding()

        return z_0, z_1


class UnetPairClassification(PairClassification):
    def get_encodings(self, features, params, mode):
        architecture = pydoc.locate(params["architecture"])
        shape = [-1] + params["input_shape"] + [1]
        x_0 = tf.reshape(features["X_0"], shape)
        x_1 = tf.reshape(features["X_1"], shape)
        enc_0 = architecture(
            x=x_0,
            training=tf.estimator.ModeKeys.TRAIN == mode,
            scope_name="unet",
            scope_reuse=False,
            params=params,
        )

        enc_1 = architecture(
            x=x_1,
            training=tf.estimator.ModeKeys.TRAIN == mode,
            scope_name="unet",
            scope_reuse=True,
            params=params
        )

        return enc_0, enc_1


class SliceClassification(EvaluateEpochsBaseTF):
    def model_fn(self, features, labels, mode, params):
        body_net = pydoc.locate(params["body_net"])
        head_net = pydoc.locate(params["head_net"])

        shape = [-1] + params["input_shape"] + [1]
        x = tf.reshape(features["X_0"], shape)
        key = params["target_label_key"]
        y = tf.reshape(features[key + "_0"], [-1])
        class_weights = params["class_weights"]

        enc = body_net(
            x=x,
            training=tf.estimator.ModeKeys.TRAIN == mode,
            scope_name="clf_body",
            scope_reuse=False
        )

        probs, preds, loss, acc = head_net(
            x=enc,
            y=y,
            class_weights=class_weights,
            n_classes=params["n_classes"]
        )

        # Make predictions
        predictions = {
            "encoding": enc,
            "classes": preds,
            "probs": probs,
            "image_label": features["image_label_0"],
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions
            )

        optimizer = tf.train.AdamOptimizer(
            learning_rate=params["learning_rate"]
        )
        train_op = optimizer.minimize(loss, tf.train.get_global_step())

        eval_metric_ops = {
            "acc": tf.metrics.mean(acc),
            "tf_metrics_acc": tf.metrics.accuracy(
                labels=y,
                predictions=preds
            ),
            "recall": tf.metrics.recall(
                labels=y,
                predictions=preds
            ),
            "specificity": tf.metrics.recall(
                labels=1 - y,
                predictions=1 - preds,
            ),
            "precision": tf.metrics.precision(
                labels=y,
                predictions=preds
            )
        }

        train_hooks = []
        log_hook_train = SumatraLoggingHook(
            ops=[loss, acc],
            names=["loss", "acc"],
            logger=self.metric_logger,
            namespace="train"
        )
        train_hooks.append(log_hook_train)

        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                training_hooks=train_hooks,
            )

        return tf.estimator.EstimatorSpec(
            mode=mode,
            eval_metric_ops=eval_metric_ops,
            loss=loss,
        )

    def gen_input_fn(self, X, y=None, mode="train", input_fn_config={}):
        return self.streamer.get_input_fn(mode)
