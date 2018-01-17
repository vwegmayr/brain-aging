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
        fc = self.fc_layer(
            tf.reshape(conv, [-1, num_features]),
            1,
            nl=tf.identity,
            name="fc",
        )
        return 63 + tf.reshape(fc, [-1], name='predictions')


def loss_guess_avg(labels):
    return tf.losses.mean_squared_error(labels, labels*0 + 63)


def model_fn(features, labels, mode, params):
    """Model function for Estimator."""
    predicted_feature = features_def.AGE

    labels = features[predicted_feature]
    m = Model(is_training=(mode == tf.estimator.ModeKeys.TRAIN))
    predictions = m.gen_output(features)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={predicted_feature: predictions}
        )

    loss = tf.losses.mean_squared_error(labels, predictions)
    loss_v_avg = loss_guess_avg(labels)

    # Calculate root mean squared error as additional eval metric
    eval_metric_ops = {
        "rmse": tf.metrics.root_mean_squared_error(
            tf.cast(labels, tf.float32),
            predictions,
        ),
        'rmse_vs_avg': tf.metrics.root_mean_squared_error(
            tf.cast(labels, tf.float32),
            predictions*0.0 + 63,
        ),
    }

    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step()
    )

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        training_hooks=[tf.train.LoggingTensorHook({
                "loss": loss,
                "loss_v_avg": loss_v_avg,
            },
            every_n_iter=10,
        )]
    )
