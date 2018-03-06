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
        self.predictions = model.gen_deconv_head(last_layer)
        self.predictions = tf.reshape(
            self.predictions,
            [-1] + mri_shape,
        )
        self.loss = tf.losses.mean_squared_error(
            self.labels,
            self.predictions,
        )
        self.metrics = {}
        super(AutoencoderHead, self).__init__(
            name=name,
            model=model,
            last_layer=last_layer,
            features=features,
            **kwargs
        )

        if len(mri_shape) == 2:
            self.image_summary_2d()
        else:
            self.image_summary_3d()

    def image_summary_2d(self):
        labels_shape = self.labels.get_shape().as_list()
        labels_for_summary = tf.reshape(
            self.labels[0],
            [1] + labels_shape[1:] + [1],
        )
        predictions_for_summary = tf.reshape(
            self.predictions[0],
            [1] + labels_shape[1:] + [1],
        )
        tf.summary.image(
            "GroundTruthXY",
            labels_for_summary,
        )
        tf.summary.image(
            "PredictionXY",
            predictions_for_summary,
        )

    def image_summary_3d(self):
        labels_shape = self.labels.get_shape().as_list()
        mid_shape = [0] + [
            self.labels.get_shape().as_list()[i] / 2
            for i in [1, 2, 3]
        ]
        labels_for_summary = tf.reshape(
            self.labels[0],
            [1] + labels_shape[1:] + [1],
        )
        predictions_for_summary = tf.reshape(
            self.predictions[0],
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

    def get_tags(self):
        tags = super(AutoencoderHead, self).get_tags()
        tags.append('autoencoder')
        return tags
