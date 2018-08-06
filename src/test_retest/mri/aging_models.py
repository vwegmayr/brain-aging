import tensorflow as tf

from src.test_retest.test_retest_base import EvaluateEpochsBaseTF
from src.baum_vagan.vagan.network_zoo.nets3D.mask_generators import unet_16_bn


class MSEDifferenceMap(EvaluateEpochsBaseTF):

    def model_fn(self, features, labels, mode, params):
        x_t0 = features["X_0"]
        x_t1 = features["X_1"]

        # reshape
        img_size = params["image_size"]
        x_t0 = tf.reshape(x_t0, [-1] + img_size + [1])
        x_t1 = tf.reshape(x_t0, [-1] + img_size + [1])

        training = mode == tf.estimator.ModeKeys.TRAIN
        training = tf.constant(training, dtype=tf.bool)

        gt_delta_x_t0 = x_t1 - x_t0
        gen_delta_x_t0 = unet_16_bn(
            x=x_t0,
            training=training
        )

        predictions = {
            "gen_delta_x_t0": gen_delta_x_t0,
            "gt_delta_x_t0": gt_delta_x_t0,
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions
            )

        loss = tf.losses.mean_squared_error(
            labels=gt_delta_x_t0,
            predictions=gen_delta_x_t0
        )

        optimizer = tf.train.AdamOptimizer(
            learning_rate=params["learning_rate"]
        )
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )

        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op
            )

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss
        )

    def gen_input_fn(self, X, y=None, mode="train", input_fn_config={}):
        return self.streamer.get_input_fn(mode)
