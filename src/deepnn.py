import tensorflow as tf
from modules.models.utils import custom_print


class DeepNN(object):
    def __init__(self):
        self.is_training = True
        self.debug_summaries = False

    def variable_summaries(self, var, name, fullcontent=True):
        """Attach a lot of summaries to a Tensor."""
        if not self.debug_summaries:
            return
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.scalar_summary('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.scalar_summary('stddev/' + name, stddev)
            tf.scalar_summary('max/' + name, tf.reduce_max(var))
            tf.scalar_summary('min/' + name, tf.reduce_min(var))
            tf.histogram_summary(name, var)

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
        with tf.variable_scope('branch_%s' % scope) as tf_scope:
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
            W_shape = filter_weights + [conv_input_shape[3], num_filters]
            W = tf.get_variable(
                "w",
                shape=[np.prod(W_shape)],
                initializer=tf.contrib.layers.xavier_initializer_conv2d(),
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
            self.variable_summaries(W, "%s/W" % scope)
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
            self.variable_summaries(
                out,
                "%s/output" % scope,
                fullcontent=False,
            )
            custom_print('%s -> [%s] -> %s' % (
                conv_input_shape, scope, out.get_shape()[1:].as_list()))
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
            self.variable_summaries(W_fc, "%s/W" % name)
            self.variable_summaries(b_fc, "%s/b" % name)
            self.variable_summaries(out, "%s/output" % name)
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
        if not self.is_train:
            return x
        return tf.nn.dropout(x, keep_prob=prob)
