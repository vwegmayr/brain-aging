import tensorflow as tf
import features as features_def
from deepnn import DeepNN
import numpy as np


class Model(DeepNN):
    def __init__(self, is_training):
        super(Model, self).__init__()
        self.is_training = is_training

    def gen_output(self, ft):
        mri = tf.cast(ft[features_def.MRI], tf.float32)
        mri = tf.reshape(mri, [-1] + mri.get_shape()[1:4].as_list() + [1])
        mri = self.batch_norm(mri, scope="norm_input")

        conv = mri
        conv = self.conv2d_shared_all_dims_layer(conv, 'b1', strides=[2, 2, 2])
        conv = self.conv2d_shared_all_dims_layer(conv, 'b2')
        conv = self.conv2d_shared_all_dims_layer(conv, 'b3')
        conv = self.conv3d_layer(conv, 24, scope="conv4")

        conv = tf.reduce_max(conv, axis=[1, 2, 3], keep_dims=True)

        num_features = np.prod(conv.get_shape().as_list()[1:])
        print '%d fc features' % (num_features)
        fc = tf.reshape(conv, [-1, num_features])
        fc = tf.concat([
            # Features from convet
            tf.reshape(conv, [-1, num_features]),
            # Additionnal features - shape [batch_size, feature_count], type float32
            #tf.reshape(tf.cast(ft[features_def.AGE], tf.float32), [-1, 1]),
        ], 1)
        fc = self.batch_norm(fc, scope="norm_ft")

        fc = self.fc_layer(
            fc,
            1,
            nl=tf.identity,
            name="fc",
        )
        return tf.reshape(fc, [-1], name='predictions')
