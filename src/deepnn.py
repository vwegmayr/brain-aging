import tensorflow as tf
from modules.models.utils import custom_print


class DeepNN(object):
    def __init__(self):
        self.is_training = True
        self.debug_summaries = False

    def variable_summaries(self, var, name, fullcontent=True):
        """Attach a lot of summaries to a Tensor."""
        with tf.name_scope('%s_summary' % name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('hist', var)

    def convet_filters_summary(self, filters, name):
        """
        input:
        @filters: [x, y, nf] shaped tensor
        """
        with tf.name_scope('%s_summary' % name):
            pass  # TODO(dhaziza)

    def conv3d(self, x, W, strides=[1, 1, 1, 1, 1], padding='VALID'):
        return tf.nn.conv3d(x, W, strides=strides, padding=padding)

    def conv2d_shared_all_dims_layer(
        self,
        _input,
        scope,
        s=5,
        num_filters_per_dim=8,
        *args,
        **kwargs
    ):
        def do_c(filters, *args, **kwargs):
            return self.conv3d_layer(
                _input,
                num_filters=num_filters_per_dim,
                filter_weights=filters,
                scope='conv',
                mpadding='SAME',
                padding='SAME',
                *args,
                **kwargs
            )
        with tf.variable_scope('%s' % scope) as tf_scope:
            b1 = do_c([s, s, 1], *args, **kwargs)
            tf_scope.reuse_variables()
            b2 = do_c([s, 1, s], *args, **kwargs)
            b3 = do_c([1, s, s], *args, **kwargs)
            return tf.concat([b1, b2, b3], 4)

    def conv3d_layer(self, x, num_filters,
                     filter_weights=[3, 3, 3],
                     nl=tf.nn.relu,
                     strides=[1, 1, 1],
                     pool=True,
                     padding='VALID',
                     mpadding='VALID',
                     scope="unnamed_conv",
                     bn=True):
        import numpy as np
        with tf.variable_scope(scope):
            conv_input_shape = x.get_shape()[1:].as_list()
            input_channels = conv_input_shape[3]
            W_shape = filter_weights + [input_channels, num_filters]
            W = tf.get_variable(
                "w",
                shape=[np.prod(filter_weights)] +
                [input_channels, num_filters],
                initializer=tf.contrib.layers.xavier_initializer(),
            )
            W = tf.reshape(W, W_shape)
            b = tf.get_variable(
                "b",
                [num_filters],
                initializer=tf.constant_initializer(0.1),
            )
            out = b + self.conv3d(
                x,
                W,
                strides=[1] + strides + [1],
                padding=padding,
            )
            if pool:
                out = tf.nn.max_pool3d(
                    out,
                    ksize=[1, 2, 2, 2, 1],
                    strides=[1, 2, 2, 2, 1],
                    padding=mpadding,
                )
            if bn:
                out = self.batch_norm(out)
            out = nl(out)
            if self.debug_summaries:
                self.variable_summaries(W, "w")
                self.variable_summaries(out, "output")
            custom_print('%s -> [%s] -> %s' % (
                conv_input_shape,
                tf.contrib.framework.get_name_scope(),
                out.get_shape()[1:].as_list()
            ))
        return out

    def fc_layer(self, x, num_outputs, nl=tf.nn.relu, name="unnamedfc"):
        with tf.variable_scope(name):
            num_inputs = x.get_shape()[1:].as_list()[0]
            W_fc = tf.get_variable(
                "w",
                shape=[num_inputs, num_outputs],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            b_fc = tf.get_variable(
                "b",
                [num_outputs],
                initializer=tf.constant_initializer(0.1)
            )
            out = nl(tf.matmul(x, W_fc) + b_fc)
            if self.debug_summaries:
                self.variable_summaries(W_fc, "W")
                self.variable_summaries(b_fc, "b")
                self.variable_summaries(out, "output")
        return out

    def batch_norm(self, x, **kwargs):
        return tf.contrib.layers.batch_norm(
            x,
            is_training=self.is_training,
            updates_collections=None,
            decay=0.9,
            **kwargs
        )

    def dropout(self, x, prob):
        if not self.is_training:
            return x
        return tf.nn.dropout(x, keep_prob=prob)
