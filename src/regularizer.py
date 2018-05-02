import tensorflow as tf


def get_batch_dimensions(X):
    d = tf.cast(tf.shape(X)[1], tf.float32)
    n = tf.cast(tf.shape(X)[0], tf.float32)

    return n, d


def l1(weights, name):
    return tf.reduce_sum(tf.abs(weights), name=name)


def l2_squared(weights, name):
    return tf.reduce_sum(tf.square(weights), name=name)


def l2(weights, name):
    return tf.sqrt(l2_squared(weights, None), name=name)


def l2_mean_batch(batch, name):
    sq_norms = tf.reduce_sum(tf.square(batch), axis=1)
    return tf.reduce_mean(sq_norms, name=name)


def icc_vals(test_batch, retest_batch):
    n_features = test_batch.get_shape()[0]


## Reliability measures for ICC computation
def MSR(Y):
    n, k = get_batch_dimensions(Y)
    df = n - 1

    mu = tf.reduce_mean(Y)
    row_means = tf.reduce_mean(Y, axis=1)
    ss = k * tf.reduce_sum(tf.square(row_means - mu))

    return ss / df


def MSW(Y):
    n, k = get_batch_dimensions(Y)
    df = n * (k - 1)

    row_means = tf.reduce_mean(Y, axis=1)
    row_means = tf.reshape(row_means, [-1, 1])
    ss = tf.reduce_sum(tf.square(Y - row_means))

    return ss / df


def MSC(Y):
    n, k = get_batch_dimensions(Y)
    df = k - 1

    mu = tf.reduce_mean(Y)
    col_means = tf.reduce_mean(Y, axis=0)
    col_means = tf.reshape(col_means, [1, -1])
    ss = n * tf.reduce_sum(tf.square(col_means - mu))

    return ss / df


def MSE(Y):
    n, k = get_batch_dimensions(Y)
    df = (n - 1) * (k - 1)

    mu = tf.reduce_mean(Y)

    col_means = tf.reduce_mean(Y, axis=0)
    col_means = tf.reshape(col_means, [1, -1])

    row_means = tf.reduce_mean(Y, axis=1)
    row_means = tf.reshape(row_means, [-1, 1])

    ss = tf.reduce_sum(tf.square(Y - col_means - row_means + mu))

    return ss / df


def fast_msr_msc_msw_mse(Y,
                         compute_msr=True,
                         compute_msc=True,
                         compute_msw=True,
                         compute_mse=True):
    n, k = get_batch_dimensions(Y)
    msr = msc = msw = mse = None

    mu = tf.reduce_mean(Y)

    col_means = tf.reduce_mean(Y, axis=0)
    col_means = tf.reshape(col_means, [1, -1])

    row_means = tf.reduce_mean(Y, axis=1)
    row_means = tf.reshape(row_means, [-1, 1])

    if compute_msr:
        msr = k * tf.reduce_sum(tf.square(row_means - mu))
        msr = msr / (n - 1)

    if compute_msc:
        msc = n * tf.reduce_sum(tf.square(col_means - mu))
        msc = msc / (k - 1)

    if compute_msw:
        msw = tf.reduce_sum(tf.square(Y - row_means))    
        msw = msw / (n * (k - 1))

    if compute_mse:
        mse = tf.reduce_sum(tf.square(Y - col_means - row_means + mu))
        mse = mse / ((n - 1) * (k - 1))

    return msr, msc, msw, mse


def ICC_C1(Y):
    n, k = get_batch_dimensions(Y)
    msr, _, _, mse = fast_msr_msc_msw_mse(Y, True, False, False, True)

    return (msr - mse) / (msr + (k - 1) * mse)


def ICC_batch_regularizer(test_batch, retest_batch, icc_func):
    """
    Compute the mean ICC over all the features.
    """
    n_features = test_batch.get_shape()[1]
    sum_icc = 0

    for i in range(n_features):
        Y_1 = tf.reshape(test_batch[:, i], [-1, 1])
        Y_2 = tf.reshape(retest_batch[:, i], [-1, 1])
        Y = tf.stack([Y_1, Y_2], axis=1)

        sum_icc += icc_func(Y)

    return sum_icc / n_features
