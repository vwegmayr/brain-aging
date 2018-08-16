import tensorflow as tf
from src.baum_vagan.tfwrapper import layers


def unet_16_bn(x, training, scope_name='generator'):

    n_ch_0 = 16

    with tf.variable_scope(scope_name):

        conv1_1 = layers.conv3D_layer_bn(x, 'conv1_1', num_filters=n_ch_0, training=training)
        conv1_2 = layers.conv3D_layer_bn(conv1_1, 'conv1_2', num_filters=n_ch_0, training=training)
        pool1 = layers.maxpool3D_layer(conv1_2)

        conv2_1 = layers.conv3D_layer_bn(pool1, 'conv2_1', num_filters=n_ch_0*2, training=training)
        conv2_2 = layers.conv3D_layer_bn(conv2_1, 'conv2_2', num_filters=n_ch_0*2, training=training)
        pool2 = layers.maxpool3D_layer(conv2_2)

        conv3_1 = layers.conv3D_layer_bn(pool2, 'conv3_1', num_filters=n_ch_0*4, training=training)
        conv3_2 = layers.conv3D_layer_bn(conv3_1, 'conv3_2', num_filters=n_ch_0*4, training=training)
        pool3 = layers.maxpool3D_layer(conv3_2)

        conv4_1 = layers.conv3D_layer_bn(pool3, 'conv4_1', num_filters=n_ch_0*8, training=training)
        conv4_2 = layers.conv3D_layer_bn(conv4_1, 'conv4_2', num_filters=n_ch_0*8, training=training)

        upconv3 = layers.deconv3D_layer_bn(conv4_2, name='upconv3', num_filters=n_ch_0, training=training)
        concat3 = layers.crop_and_concat_layer_fixed([upconv3, conv3_2], axis=-1)

        conv5_1 = layers.conv3D_layer_bn(concat3, 'conv5_1', num_filters=n_ch_0*4, training=training)

        conv5_2 = layers.conv3D_layer_bn(conv5_1, 'conv5_2', num_filters=n_ch_0*4, training=training)

        upconv2 = layers.deconv3D_layer_bn(conv5_2, name='upconv2', num_filters=n_ch_0, training=training)
        concat2 = layers.crop_and_concat_layer_fixed([upconv2, conv2_2], axis=-1)

        conv6_1 = layers.conv3D_layer_bn(concat2, 'conv6_1', num_filters=n_ch_0*2, training=training)
        conv6_2 = layers.conv3D_layer_bn(conv6_1, 'conv6_2', num_filters=n_ch_0*2, training=training)

        upconv1 = layers.deconv3D_layer_bn(conv6_2, name='upconv1', num_filters=n_ch_0, training=training)
  
        concat1 = layers.crop_and_concat_layer_fixed([upconv1, conv1_2], axis=-1)
        #concat1 = upconv1

        conv8_1 = layers.conv3D_layer_bn(concat1, 'conv8_1', num_filters=n_ch_0, training=training)
        conv8_2 = layers.conv3D_layer(conv8_1, 'conv8_2', num_filters=1, activation=tf.identity)

    return conv8_2


def unet_16_bn_allow_reuse(x, training, scope_name='generator', scope_reuse=True):

    n_ch_0 = 16

    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):

        conv1_1 = layers.conv3D_layer_bn(x, 'conv1_1', num_filters=n_ch_0, training=training)
        conv1_2 = layers.conv3D_layer_bn(conv1_1, 'conv1_2', num_filters=n_ch_0, training=training)
        pool1 = layers.maxpool3D_layer(conv1_2)

        conv2_1 = layers.conv3D_layer_bn(pool1, 'conv2_1', num_filters=n_ch_0*2, training=training)
        conv2_2 = layers.conv3D_layer_bn(conv2_1, 'conv2_2', num_filters=n_ch_0*2, training=training)
        pool2 = layers.maxpool3D_layer(conv2_2)

        conv3_1 = layers.conv3D_layer_bn(pool2, 'conv3_1', num_filters=n_ch_0*4, training=training)
        conv3_2 = layers.conv3D_layer_bn(conv3_1, 'conv3_2', num_filters=n_ch_0*4, training=training)
        pool3 = layers.maxpool3D_layer(conv3_2)

        conv4_1 = layers.conv3D_layer_bn(pool3, 'conv4_1', num_filters=n_ch_0*8, training=training)
        conv4_2 = layers.conv3D_layer_bn(conv4_1, 'conv4_2', num_filters=n_ch_0*8, training=training)

        upconv3 = layers.deconv3D_layer_bn(conv4_2, name='upconv3', num_filters=n_ch_0, training=training)
        concat3 = layers.crop_and_concat_layer_fixed([upconv3, conv3_2], axis=-1)

        conv5_1 = layers.conv3D_layer_bn(concat3, 'conv5_1', num_filters=n_ch_0*4, training=training)

        conv5_2 = layers.conv3D_layer_bn(conv5_1, 'conv5_2', num_filters=n_ch_0*4, training=training)

        upconv2 = layers.deconv3D_layer_bn(conv5_2, name='upconv2', num_filters=n_ch_0, training=training)
        concat2 = layers.crop_and_concat_layer_fixed([upconv2, conv2_2], axis=-1)

        conv6_1 = layers.conv3D_layer_bn(concat2, 'conv6_1', num_filters=n_ch_0*2, training=training)
        conv6_2 = layers.conv3D_layer_bn(conv6_1, 'conv6_2', num_filters=n_ch_0*2, training=training)

        upconv1 = layers.deconv3D_layer_bn(conv6_2, name='upconv1', num_filters=n_ch_0, training=training)
  
        concat1 = layers.crop_and_concat_layer_fixed([conv1_2, upconv1], axis=-1)
        #concat1 = upconv1

        conv8_1 = layers.conv3D_layer_bn(concat1, 'conv8_1', num_filters=n_ch_0, training=training)
        conv8_2 = layers.conv3D_layer(conv8_1, 'conv8_2', num_filters=1, activation=tf.identity)

    return conv8_2


def unet_16_bn_iterated(x, training, scope_name='generator', max_iterations=3):
    """
    Second channel should contain the number of iterations.
    """
    delta_im = x[:, :, :, :, 1:2]
    delta = delta_im[0, 0, 0, 0, 0]

    n_channels = x.get_shape().as_list()[-1]
    splits = tf.split(x, n_channels, axis=-1)

    x = tf.concat(splits[0:1] + splits[2:], axis=-1)

    # create weights
    unet_16_bn_allow_reuse(
        x,
        training,
        scope_name=scope_name,
        scope_reuse=False
    )

    def iterate(inp, n_steps):
        out = inp
        for i in range(n_steps):
            out = unet_16_bn_allow_reuse(
                out,
                training,
                scope_name=scope_name,
            )

        return out

    iterations = [x]
    for i in range(1, max_iterations + 1):
        last = iterations[i - 1]
        iterations.append(iterate(last, 1))

    conditions = [None for i in range(max_iterations)]
    conditions[max_iterations - 1] = tf.cond(
        tf.equal(delta, max_iterations - 1),
        lambda: iterations[max_iterations - 1],
        lambda: iterations[max_iterations]
    )

    for i in range(max_iterations - 2, 0, -1):
        cond = tf.cond(
            tf.equal(delta, i),
            lambda: iterations[i],
            lambda: conditions[i + 1]
        )
        conditions[i] = cond

    res = conditions[1]

    return res
