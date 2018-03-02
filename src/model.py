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

        def conv_relu(input_, kernel_shape, scope):
            return self.conv3d_layer(
                input_,
                kernel_shape[4],
                kernel_shape[0:3],
                pool=False,
                strides=[2, 2, 2],
                bn=False,
                scope=scope,
                padding='SAME',
            )

        conv = mri
        conv = tf.concat([
            conv_relu(conv, [5, 5, 5, -1, 15], scope='conv1_a'),
            conv_relu(conv, [6, 6, 6, -1, 15], scope='conv1_b'),
            conv_relu(conv, [7, 7, 7, -1, 15], scope='conv1_c'),
        ], 4)
        conv = conv_relu(conv, kernel_shape=[5, 5, 5, 45, 60], scope='conv2')
        conv = conv_relu(conv, kernel_shape=[5, 5, 5, 60, 64], scope='conv3')
        conv = conv_relu(conv, kernel_shape=[3, 3, 3, 64, 100], scope='conv4')
        conv = conv_relu(conv, kernel_shape=[3, 3, 3, 100, 128], scope='conv5')
        conv = conv_relu(conv, kernel_shape=[3, 3, 3, 128, 256], scope='conv6')
        conv = conv_relu(conv, kernel_shape=[3, 3, 3, 256, 512], scope='conv7')
        return tf.verify_tensor_all_finite(conv, "gen_last_layer returns non finite values!")

    def gen_head(self, fc, num_classes, **kwargs):
        assert(fc.get_shape().as_list()[1:4] == [1, 1, 1])
        ft_in = fc.get_shape().as_list()[4]
        fc = tf.reshape(fc, [tf.shape(fc)[0], ft_in])
        fc = self.fc_layer(fc, 512, name='fullcn')
        fc = self.dropout(fc, 0.3)
        fc = self.fc_layer(fc, num_classes, nl=tf.identity, name='logits')
        return tf.verify_tensor_all_finite(fc, "gen_head returns non finite values!")

    def gen_deconv_head(self, fc):
        assert(len(self.cnn_layers_shapes) > 0)
        fc = self.batch_norm(fc, scope='ft_norm')
        in_ft = fc.get_shape().as_list()[-1]
        ft_count = [256, 100, 64, 64, 64, 32]
        assert(len(ft_count) == len(self.cnn_layers_shapes))

        # Handle first layer manually - manual broadcast if needed
        conv = fc
        if len(conv.get_shape()) == 2:
            conv = tf.reshape(conv, [tf.shape(conv)[0], 1, 1, 1, in_ft])
            first_shape = copy.copy(self.cnn_layers_shapes[-1]['shape'])
            first_shape[-1] = ft_count[0]
            first_shape[0] = tf.shape(fc)[0]  # batch size
            conv = tf.multiply(tf.ones(first_shape), conv)

        assert(len(conv.get_shape()) == 5)

        non_linearities = [tf.nn.relu] * len(self.cnn_layers_shapes)

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
                filter_weights=[3, 3, 3],
            )
        conv = self.conv3d_layer(
            conv,
            1,
            [1, 1, 1],
            pool=False,
            bn=False,
            strides=[1, 1, 1],
            scope='reconstructed',
            mpadding='SAME',
            padding='SAME',
            nl=tf.identity,
        )
        return conv
