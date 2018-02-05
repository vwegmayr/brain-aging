import tensorflow as tf
from modules.models.utils import custom_print
import features as features_def
from deepnn import DeepNN
import numpy as np


class Model(DeepNN):
    def __init__(self, is_training):
        super(Model, self).__init__()
        self.is_training = is_training

    def gen_last_layer(self, ft):
        mri = tf.cast(ft[features_def.MRI], tf.float32)
        mri = tf.reshape(mri, [-1] + mri.get_shape()[1:4].as_list() + [1])
        mri = self.batch_norm(mri, scope="norm_input")

        def conv_wrap(conv, filters, size, scope, pool=True):
            return self.conv3d_layer(
                conv,
                filters,
                size,
                pool=pool,
                bn=True,
                scope=scope,
                mpadding='SAME',
                padding='SAME',
            )

        conv = mri
        conv = self.conv2d_shared_all_dims_layer(conv, 'b1')
        conv = conv_wrap(conv, 60, [5, 5, 5], "c2")
        conv = conv_wrap(conv, 60, [5, 5, 5], "c3")
        conv = conv_wrap(conv, 100, [3, 3, 3], "c4")
        conv = conv_wrap(conv, 100, [3, 3, 3], "c5")

        conv = tf.reduce_max(conv, axis=[1, 2, 3])

        custom_print('%d fc features' % (conv.get_shape().as_list()[1]))
        fc = tf.concat([
            # Features from convet
            conv,
            # Additionnal features:
            #    shape [batch_size, feature_count]
            #    type float32
            # tf.reshape(tf.cast(ft[features_def.AGE], tf.float32), [-1, 1]),
        ], 1)

        fc = self.fc_layer(
            fc,
            256,
            name="fc_features",
        )
        output = self.batch_norm(fc, scope='ft_norm')

        # Summaries:
        with tf.variable_scope("b1/conv", reuse=True):
            self.convet_filters_summary(
                tf.reshape(
                    tf.get_variable('w'),
                    [5, 5, 1, -1],
                ),
                "Conv2D"
            )
        return output

    def gen_head_regressor(self, last_layer, predicted_avg):
        return predicted_avg + self.fc_layer(
            last_layer,
            len(predicted_avg),
            nl=tf.identity,
            name="fc_regressor",
        )

    def gen_head_classifier(self, last_layer, num_classes):
        return self.fc_layer(
            last_layer,
            num_classes,
            name="fc_classifier",
        )
