
import tensorflow as tf
from src.baum_vagan.tfwrapper import layers


def C3D_fcn_16_2D(x, training, scope_name='critic', scope_reuse=False):

    with tf.variable_scope(scope_name) as scope:
        if scope_reuse:
            scope.reuse_variables()

        conv1_1 = layers.conv2D_layer(x, 'conv1_1', num_filters=16)

        pool1 = layers.maxpool2D_layer(conv1_1)

        conv2_1 = layers.conv2D_layer(pool1, 'conv2_1', num_filters=32)

        pool2 = layers.maxpool2D_layer(conv2_1)

        conv3_1 = layers.conv2D_layer(pool2, 'conv3_1', num_filters=64)
        conv3_2 = layers.conv2D_layer(conv3_1, 'conv3_2', num_filters=64)

        pool3 = layers.maxpool2D_layer(conv3_2)

        conv4_1 = layers.conv2D_layer(pool3, 'conv4_1', num_filters=128)
        conv4_2 = layers.conv2D_layer(conv4_1, 'conv4_2', num_filters=128)

        pool4 = layers.maxpool2D_layer(conv4_2)

        conv5_1 = layers.conv2D_layer(pool4, 'conv5_1', num_filters=256)
        conv5_2 = layers.conv2D_layer(conv5_1, 'conv5_2', num_filters=256)

        convD_1 = layers.conv2D_layer(conv5_2, 'convD_1', num_filters=256)
        convD_2 = layers.conv2D_layer(convD_1,
                                         'convD_2',
                                         num_filters=1,
                                         kernel_size=(1,1,1),
                                         activation=tf.identity)

        logits = layers.averagepool2D_layer(convD_2, name='diagnosis_avg')

    return logits


def C3D_fcn_16_2D_body(x, training, scope_name='critic', scope_reuse=False):

    with tf.variable_scope(scope_name) as scope:
        if scope_reuse:
            scope.reuse_variables()

        conv1_1 = layers.conv2D_layer(x, 'conv1_1', num_filters=16)

        pool1 = layers.maxpool2D_layer(conv1_1)

        conv2_1 = layers.conv2D_layer(pool1, 'conv2_1', num_filters=32)

        pool2 = layers.maxpool2D_layer(conv2_1)

        conv3_1 = layers.conv2D_layer(pool2, 'conv3_1', num_filters=64)
        conv3_2 = layers.conv2D_layer(conv3_1, 'conv3_2', num_filters=64)

        pool3 = layers.maxpool2D_layer(conv3_2)

    return pool3


def C3D_fcn_16_2D_head(x, training, scope_name='critic', scope_reuse=False):

    with tf.variable_scope(scope_name) as scope:
        if scope_reuse:
            scope.reuse_variables()

        conv4_1 = layers.conv2D_layer(x, 'conv4_1', num_filters=128)
        conv4_2 = layers.conv2D_layer(conv4_1, 'conv4_2', num_filters=128)

        pool4 = layers.maxpool2D_layer(conv4_2)

        conv5_1 = layers.conv2D_layer(pool4, 'conv5_1', num_filters=256)
        conv5_2 = layers.conv2D_layer(conv5_1, 'conv5_2', num_filters=256)

        convD_1 = layers.conv2D_layer(conv5_2, 'convD_1', num_filters=256)
        convD_2 = layers.conv2D_layer(convD_1,
                                         'convD_2',
                                         num_filters=1,
                                         kernel_size=(1,1,1),
                                         activation=tf.identity)

        logits = layers.averagepool2D_layer(convD_2, name='diagnosis_avg')

    return logits


def C3D_fcn_16_2D_conditioned_with_delta_head_sharing(
        x, training, scope_name='critic', scope_reuse=False):

    delta_x0 = x[:, :, :, 2:3]
    delta_img = x[:, :, :, 1:2]
    delta = delta_img[:, 0, 0, :]
    x0 = x[:, :, :, 0:1]

    x0_body = C3D_fcn_16_2D_body(
        x0,
        training,
        scope_name="critic_x0_body",
        scope_reuse=scope_reuse
    )

    delta_x0_body = C3D_fcn_16_2D_body(
        delta_x0,
        training,
        scope_name="critic_delta_x0_body",
        scope_reuse=scope_reuse
    )

    conc = tf.concat([x0_body, delta_x0_body], axis=-1)

    logits = C3D_fcn_16_2D_head(
        conc,
        training,
        scope_name="critic_head",
        scope_reuse=scope_reuse
    )

    all_logits = tf.concat([logits, delta], axis=-1)
    decision = tf.layers.dense(all_logits, 1, reuse=scope_reuse)

    return decision


def C3D_fcn_16_2D_conditioned(x, training, scope_name='critic', scope_reuse=False):

    with tf.variable_scope(scope_name) as scope:
        if scope_reuse:
            scope.reuse_variables()

        # check if last channel larger than 1
        condition = x[:, :, :, 1:2]
        x = x[:, :, :, 0:1]

        conv1_1 = layers.conv2D_layer(x, 'conv1_1', num_filters=16)

        pool1 = layers.maxpool2D_layer(conv1_1)

        conv2_1 = layers.conv2D_layer(pool1, 'conv2_1', num_filters=32)

        pool2 = layers.maxpool2D_layer(conv2_1)

        conv3_1 = layers.conv2D_layer(pool2, 'conv3_1', num_filters=64)
        conv3_2 = layers.conv2D_layer(conv3_1, 'conv3_2', num_filters=64)

        pool3 = layers.maxpool2D_layer(conv3_2)

        conv4_1 = layers.conv2D_layer(pool3, 'conv4_1', num_filters=128)
        conv4_2 = layers.conv2D_layer(conv4_1, 'conv4_2', num_filters=128)

        pool4 = layers.maxpool2D_layer(conv4_2)

        conv5_1 = layers.conv2D_layer(pool4, 'conv5_1', num_filters=256)
        conv5_2 = layers.conv2D_layer(conv5_1, 'conv5_2', num_filters=256)

        convD_1 = layers.conv2D_layer(conv5_2, 'convD_1', num_filters=256)
        convD_2 = layers.conv2D_layer(convD_1,
                                         'convD_2',
                                         num_filters=1,
                                         kernel_size=(1,1,1),
                                         activation=tf.identity)

        logits = layers.averagepool2D_layer(convD_2, name='diagnosis_avg')

        condition_logits = C3D_fcn_16_2D(
            condition, training, scope_name="critic_condition", scope_reuse=scope_reuse
        )

        all_logits = tf.concat([logits, condition_logits], axis=-1)
        decision = tf.layers.dense(all_logits, 1, reuse=scope_reuse)

    return decision


def C3D_fcn_16_2D_conditioned_with_delta(x, training, scope_name='critic', scope_reuse=False):

    with tf.variable_scope(scope_name) as scope:
        if scope_reuse:
            scope.reuse_variables()

        # discrimnator is conditioned on the difference map
        condition = x[:, :, :, 2:3]
        delta_img = x[:, :, : 1:2]
        delta = delta_img[:, 0, 0, :]
        x = x[:, :, :, 0:1]

        conv1_1 = layers.conv2D_layer(x, 'conv1_1', num_filters=16)

        pool1 = layers.maxpool2D_layer(conv1_1)

        conv2_1 = layers.conv2D_layer(pool1, 'conv2_1', num_filters=32)

        pool2 = layers.maxpool2D_layer(conv2_1)

        conv3_1 = layers.conv2D_layer(pool2, 'conv3_1', num_filters=64)
        conv3_2 = layers.conv2D_layer(conv3_1, 'conv3_2', num_filters=64)

        pool3 = layers.maxpool2D_layer(conv3_2)

        conv4_1 = layers.conv2D_layer(pool3, 'conv4_1', num_filters=128)
        conv4_2 = layers.conv2D_layer(conv4_1, 'conv4_2', num_filters=128)

        pool4 = layers.maxpool2D_layer(conv4_2)

        conv5_1 = layers.conv2D_layer(pool4, 'conv5_1', num_filters=256)
        conv5_2 = layers.conv2D_layer(conv5_1, 'conv5_2', num_filters=256)

        convD_1 = layers.conv2D_layer(conv5_2, 'convD_1', num_filters=256)
        convD_2 = layers.conv2D_layer(convD_1,
                                         'convD_2',
                                         num_filters=1,
                                         kernel_size=(1,1,1),
                                         activation=tf.identity)

        logits = layers.averagepool2D_layer(convD_2, name='diagnosis_avg')

        condition_logits = C3D_fcn_16_2D(
            condition, training, scope_name="critic_condition", scope_reuse=scope_reuse
        )

        all_logits = tf.concat([logits, condition_logits, delta], axis=-1)
        decision = tf.layers.dense(all_logits, 1, reuse=scope_reuse)

    return decision


def C3D_fcn_16_2D_conditioned_with_delta_channel(x, training, scope_name='critic', scope_reuse=False):

    delta_x0_with_delta = x[:, :, :, 1:3]
    x0 = x[:, :, :, 0:1]

    x0_logits = C3D_fcn_16_2D(
        x0,
        training,
        scope_name="critic_x0",
        scope_reuse=scope_reuse
    )

    delta_x0_logits = C3D_fcn_16_2D(
        delta_x0_with_delta,
        training,
        scope_name="critic_delta_x0",
        scope_reuse=scope_reuse
    )

    all_logits = tf.concat([x0_logits, delta_x0_logits], axis=-1)
    out = tf.layers.dense(all_logits, 1, reuse=scope_reuse)

    return out


def C3D_fcn_16_2D_bn(x, training, scope_name='critic', scope_reuse=False):

    with tf.variable_scope(scope_name) as scope:
        if scope_reuse:
            scope.reuse_variables()

        conv1_1 = layers.conv2D_layer_bn(x, 'conv1_1', num_filters=16, training=training)

        pool1 = layers.maxpool2D_layer(conv1_1)

        conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=32, training=training)

        pool2 = layers.maxpool2D_layer(conv2_1)

        conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=64, training=training)
        conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=64, training=training)

        pool3 = layers.maxpool2D_layer(conv3_2)

        conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=128, training=training)
        conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=128,training=training)

        pool4 = layers.maxpool2D_layer(conv4_2)

        conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=256, training=training)
        conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=256, training=training)

        convD_1 = layers.conv2D_layer_bn(conv5_2, 'convD_1', num_filters=256, training=training)
        convD_2 = layers.conv2D_layer(convD_1,
                                         'convD_2',
                                         num_filters=1,
                                         kernel_size=(1,1,1),
                                         activation=tf.identity)

        logits = layers.averagepool2D_layer(convD_2, name='diagnosis_avg')

    return logits