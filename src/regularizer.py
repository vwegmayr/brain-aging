import tensorflow as tf


def l1(weights, name):
    return tf.reduce_sum(tf.abs(weights), name=name)


def l2_squared(weights, name):
    return tf.reduce_sum(tf.square(weights), name=name)


def l2(weights, name):
    return tf.sqrt(l2_squared(weights), name=name)


def l2_mean_batch(batch, name):
    sq_norms = tf.reduce_sum(tf.square(batch), axis=1)
    return tf.reduce_mean(sq_norms, name=name)
