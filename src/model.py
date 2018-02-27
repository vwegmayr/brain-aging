import copy
import tensorflow as tf
import features as features_def
from deepnn import DeepNN


class Model(DeepNN):
    def __init__(self, is_training, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.is_training = is_training

    def gen_last_layer(self, ft):
        mri = tf.cast(ft[features_def.MRI], tf.float32)
        mri = tf.reshape(mri, [-1] + mri.get_shape()[1:4].as_list() + [1])
        mri = self.batch_norm(mri, scope="norm_input")

        def conv_wrap(conv, filters, size, scope, pool=True):
            out = self.conv3d_layer(
                conv,
                filters,
                size,
                pool=pool,
                bn=True,
                scope=scope,
                mpadding='SAME',
                padding='SAME',
            )
            self.on_cnn_layer(out)
            return out

        conv = mri
        self.on_cnn_layer(conv, "input")
        conv = self.conv2d_shared_all_dims_layer(conv, 'b1')
        self.on_cnn_layer(conv)
        conv = conv_wrap(conv, 60, [5, 5, 5], "c2")
        conv = conv_wrap(conv, 60, [5, 5, 5], "c3")
        conv = conv_wrap(conv, 100, [3, 3, 3], "c4")
        conv = conv_wrap(conv, 100, [3, 3, 3], "c5")

        conv = tf.reduce_max(conv, axis=[1, 2, 3])

        self.print_shape('%d fc features' % (conv.get_shape().as_list()[1]))
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

        # Summaries:
        with tf.variable_scope("b1/conv", reuse=True):
            self.convet_filters_summary(
                tf.reshape(
                    tf.get_variable('w'),
                    [5, 5, 1, -1],
                ),
                "Conv2D"
            )
        return fc

    def gen_head(self, fc, num_classes, **kwargs):
        fc = self.batch_norm(fc, scope='ft_norm')
        fc = self.fc_layer(
            fc,
            256,
            name="fc_head1",
        )
        fc = self.batch_norm(fc, scope='ft_norm2')
        fc = self.fc_layer(
            fc,
            num_classes,
            name="fc_head2",
            **kwargs
        )
        return fc

    def gen_deconv_head(self, fc):
        assert(len(self.cnn_layers_shapes) > 0)
        assert(len(fc.get_shape()) == 2)
        fc = self.batch_norm(fc, scope='ft_norm')
        in_ft = fc.get_shape().as_list()[1]
        conv = tf.reshape(fc, [tf.shape(fc)[0], 1, 1, 1, in_ft])
        ft_count = [512, 256, 256, 128, 128, 64, 1]
        assert(len(ft_count) == len(self.cnn_layers_shapes))

        # Handle first layer manually - manual broadcast
        first_shape = copy.copy(self.cnn_layers_shapes[-1]['shape'])
        first_shape[-1] = ft_count[0]
        first_shape[0] = tf.shape(fc)[0]
        conv = tf.ones(first_shape) * conv

        non_linearities = [tf.nn.relu] * len(self.cnn_layers_shapes)
        non_linearities[-1] = tf.identity
        non_linearities[-2] = tf.nn.elu

        for layer, num_filters, nl in zip(
            reversed(self.cnn_layers_shapes),
            ft_count,
            non_linearities,
        )[1:]:
            output_shape = copy.copy(layer['shape'])
            output_shape[-1] = num_filters
            # Compute strides
            prev_shape = conv.get_shape().as_list()
            strides = int(round(float(output_shape[1]) / prev_shape[1]))
            conv = self.conv3d_layer_transpose(
                conv,
                num_filters=num_filters,
                output_shape=output_shape,
                scope=layer['name'],
                strides=[strides] * 3,
                nl=nl,
            )
        return conv
