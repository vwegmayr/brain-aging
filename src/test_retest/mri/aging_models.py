import tensorflow as tf
import os

from src.test_retest.test_retest_base import EvaluateEpochsBaseTF
from src.baum_vagan.vagan.network_zoo.nets3D.mask_generators import unet_16_bn
from src.baum_vagan.vagan.network_zoo.nets2D.mask_generators import unet_16_2D_bn
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

        if len(img_size) == 2:
            mode_2D = True
            gen_net = unet_16_2D_bn
        else:
            mode_2D = False
            gen_net = unet_16_bn

        gt_delta_x_t0 = x_t1 - x_t0
        gen_delta_x_t0 = gen_net(
            x=x_t0,
            training=training
        )

        if params["use_tanh"]:
            gen_delta_x_t0 = tf.tanh(gen_delta_x_t0)

        l1_map_reg = tf.reduce_mean(tf.abs(gen_delta_x_t0))
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

        loss += l1_map_reg * params["l1_map_weight"]

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

        if mode_2D:
            disp_gt_delta_x_t0 = gt_delta_x_t0
            disp_x_t0 = x_t0
            disp_gen_delta_x_t0 = gen_delta_x_t0
            disp_gen_x_t1 = gen_x_t1
        else:
            disp_gt_delta_x_t0 = gt_delta_x_t0[:, :, :, z_slice, :]
            disp_x_t0 = x_t0[:, :, :, z_slice, :]
            disp_gen_delta_x_t0 = gen_delta_x_t0[:, :, :, z_slice, :]
            disp_gen_x_t1 = gen_x_t1[:, :, :, z_slice, :]

        gt_summ = tf.summary.image(
            name='difference_map_gt',
            tensor=tf.abs(disp_gt_delta_x_t0),
            max_outputs=8
        )

        diff_gen_summ = tf.summary.image(
            name="difference_map_gen",
            tensor=tf.abs(disp_gen_delta_x_t0),
            max_outputs=8
        )

        x_t0_summ = tf.summary.image(
            name='x_t0',
            tensor=disp_x_t0,
            max_outputs=8
        )

        gen_x_t1_summ = tf.summary.image(
            name="x_t1_gen",
            tensor=disp_gen_x_t1,
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
                training_hooks=train_hooks
            )

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            evaluation_hooks=eval_hooks
        )

    def gen_input_fn(self, X, y=None, mode="train", input_fn_config={}):
        return self.streamer.get_input_fn(mode)


class SyntheticMSEDifferenceMap(MSEDifferenceMap):
    def gen_input_fn(self, X, y=None, mode='train', input_fn_config={}):
        return self.streamer.get_input_fn(mode)
