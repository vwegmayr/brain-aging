import tensorflow as tf


from src.test_retest.test_retest_base import EvaluateEpochsBaseTF
from .model_components import MultiLayerPairEncoder, \
    PairClassificationHead
from src.train_hooks import SumatraLoggingHook, HookFactory


class PairClassification(EvaluateEpochsBaseTF):
    def model_fn(self, features, labels, mode, params):
        encoder = MultiLayerPairEncoder(
            features=features,
            params=params,
            streamer=self.streamer
        )

        enc_0, enc_1 = encoder.get_encodings()

        clf = PairClassificationHead(
            features=features,
            params=params,
            encoder=encoder
        )

        preds_0, preds_1 = clf.get_predictions()

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

        train_hook_names = params["train_hooks"]
        eval_hook_names = params["eval_hooks"]

        hf = HookFactory(
            streamer=self.streamer,
            logger=self.metric_logger,
            out_dir=self.data_params["dump_out_dir"],
            model_save_path=self.save_path,
            epoch=self.current_epoch
        )

        if "embeddings" in train_hook_names:
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

            # Robustness of head predictions
            hook = hf.get_tensor_prediction_robustness_hook(
                [preds_0, preds_1],
                [features["file_name_0"], features["file_name_1"]],
                "head_prediction",
                True
            )
            train_hooks.append(hook)

            hook = hf.get_tensor_prediction_robustness_hook(
                [preds_0, preds_1],
                [features["file_name_0"], features["file_name_1"]],
                "head_prediction",
                False
            )
            eval_hooks.append(hook)

        if "predictions" in eval_hook_names:
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

        log_hook_test = SumatraLoggingHook(
            ops=[loss, acc] + all_losses,
            names=["loss", "acc"] + loss_names,
            logger=self.metric_logger,
            namespace="test"
        )
        eval_hooks.append(log_hook_test)

        if self.current_epoch == self.n_epochs - 1:
            eval_hooks.append(hf.get_file_summarizer_hook(
                ["prediction_robustness", "predictions",
                 "head_prediction"]
            ))

        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                training_hooks=train_hooks
            )

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            evaluation_hooks=eval_hooks
        )

    def gen_input_fn(self, X, y=None, train=True, input_fn_config={}):
        return self.streamer.get_input_fn(train)
