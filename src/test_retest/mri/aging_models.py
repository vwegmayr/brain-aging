import tensorflow as tf
import os

from src.test_retest.test_retest_base import EvaluateEpochsBaseTF
from src.baum_vagan.vagan.network_zoo.nets3D.mask_generators import unet_16_bn
from src.baum_vagan.tfwrapper.utils import put_kernels_on_grid


class MSEDifferenceMap(EvaluateEpochsBaseTF):

    def model_fn(self, features, labels, mode, params):
        x_t0 = features["X_0"]
        x_t1 = features["X_1"]
        batch_size = tf.shape(x_t0)[0]
        # reshape
        img_size = params["image_size"]
        x_t0 = tf.reshape(x_t0, [-1] + img_size + [1])
        x_t1 = tf.reshape(x_t1, [-1] + img_size + [1])

        training = mode == tf.estimator.ModeKeys.TRAIN
        # training = tf.constant(training, dtype=tf.bool)

        gt_delta_x_t0 = x_t1 - x_t0
        gen_delta_x_t0 = unet_16_bn(
            x=x_t0,
            training=training
        )
        gen_x_t1 = x_t0 + gen_delta_x_t0

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

        # some summaries
        if training:
            img_summ_dir = os.path.join(self.save_path, "train")
        else:
            img_summ_dir = os.path.join(self.save_path, "eval")

        z_slice = params["z_slice"]
        gt_summ = tf.summary.image(
            name='difference_map_gt',
            tensor=gt_delta_x_t0[:, :, :, z_slice, :],
            max_outputs=8
        )

        x_t0_summ = tf.summary.image(
            name='x_t0',
            tensor=x_t0[:, :, :, z_slice, :],
            max_outputs=8
        )

        diff_gen_summ = tf.summary.image(
            name="difference_map_gen",
            tensor=gen_delta_x_t0[:, :, :, z_slice, :],
            max_outputs=8
        )

        gen_x_t1_summ = tf.summary.image(
            name="x_t1_gen",
            tensor=gen_x_t1[:, :, :, z_slice, :],
            max_outputs=8
        )

        img_summary_op = tf.summary.merge([
            gt_summ, x_t0_summ, diff_gen_summ, gen_x_t1_summ
        ])
        img_summary_hook = tf.train.SummarySaverHook(
            save_steps=1,
            summary_op=img_summary_op,
            output_dir=img_summ_dir
        )

        train_hooks = [img_summary_hook]
        eval_hooks = [img_summary_hook]

        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                #training_hooks=train_hooks
            )

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            #evaluation_hooks=eval_hooks
        )

    def gen_input_fn(self, X, y=None, mode="train", input_fn_config={}):
        return self.streamer.get_input_fn(mode)
