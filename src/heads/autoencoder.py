import tensorflow as tf
from base import NetworkHeadBase
import src.features as ft_def


class AutoencoderHead(NetworkHeadBase):
    def __init__(
        self,
        # NetworkHeadBase arguments
        name,
        model,
        last_layer,
        features,
        # Custom arguments (from config file)
        # ...
        # Args passed to parent
        **kwargs
    ):
        self.labels = features[ft_def.MRI]
        mri_shape = self.labels.get_shape().as_list()[1:]
        predictions = tf.reshape(
            model.gen_deconv_head(last_layer),
            [-1] + mri_shape,
        )
        mri_avg = tf.get_variable(
            "mri_avg",
            shape=mri_shape,
            initializer=tf.zeros_initializer(),
        )
        mri_mult_factor = tf.get_variable(
            "mri_mult_factor",
            shape=mri_shape,
            initializer=tf.zeros_initializer(),
        )
        self.predictions = mri_avg + tf.multiply(predictions, mri_mult_factor)
        self.loss = tf.losses.mean_squared_error(
            self.labels,
            self.predictions,
        )
        super(AutoencoderHead, self).__init__(
            name=name,
            model=model,
            last_layer=last_layer,
            features=features,
            **kwargs
        )

        self.image_summary()

    def image_summary(self):
        labels_shape = self.labels.get_shape().as_list()
        mid_shape = [0] + [
            self.labels.get_shape().as_list()[i] / 2
            for i in [1, 2, 3]
        ]
        labels_for_summary = tf.reshape(
            tf.cast(self.labels[0], tf.float32),
            [1] + labels_shape[1:] + [1],
        )
        predictions_for_summary = tf.reshape(
            tf.cast(self.predictions[0], tf.float32),
            [1] + labels_shape[1:] + [1],
        )
        tf.summary.image(
            "GroundTruthXY",
            labels_for_summary[:, :, :, mid_shape[3]],
        )
        tf.summary.image(
            "PredictionXY",
            predictions_for_summary[:, :, :, mid_shape[3]],
        )
        tf.summary.image(
            "GroundTruthXZ",
            labels_for_summary[:, :, mid_shape[2], :],
        )
        tf.summary.image(
            "PredictionXZ",
            predictions_for_summary[:, :, mid_shape[2], :],
        )
        tf.summary.image(
            "GroundTruthYZ",
            labels_for_summary[:, mid_shape[1], :, :],
        )
        tf.summary.image(
            "PredictionYZ",
            predictions_for_summary[:, mid_shape[1], :, :],
        )
