import tensorflow as tf


# Labels for regularizers
JS_DIVERGENCE_LABEL = "js_divergence"
L2_SQUARED_LABEL = "l2_sq"
COSINE_SIMILARITY = "cosine_sim"
L1_MEAN = "l1_mean"


# Traditional norm regularizers
def l1(weights, name):
    return tf.reduce_sum(tf.abs(weights), name=name)


def l1_mean(x):
    return tf.reduce_mean(tf.abs(x))


def l2_squared(weights, name):
    return tf.reduce_sum(tf.square(weights), name=name)


def l2_squared_mean_batch(weights, name):
    return tf.reduce_mean(tf.reduce_sum(tf.square(weights), axis=1), name=name)


def l2(weights, name):
    return tf.sqrt(l2_squared(weights, None), name=name)


def l2_mean_batch(batch, name):
    sq_norms = tf.reduce_sum(tf.square(batch), axis=1)
    return tf.reduce_mean(sq_norms, name=name)


# Distribution divergence measures
def kl_divergence(p, q, eps=0.000001):
    """
    Kullback-Leibler divergence.

    Args:
        - p: vector containing P(i) in its i-th component
        - q: vector containing Q(i) in its i-th component

    Return:
        - kl: KL-divergence between p and q
    """
    # Add small value to p to avoid division by 0
    c_p = p + eps
    c_p = c_p / tf.reduce_sum(c_p)

    # Add small value to q to avoid log of 0
    c_q = q + eps
    c_q = c_q / tf.reduce_sum(c_q)

    div = c_q / c_p
    kl = - tf.reduce_sum(c_p * tf.log(div))
    return kl


def js_divergence(p, q, eps=0.000001):
    """
    Jensen-Shannon divergence.

    Args:
        - p: vector containing P(i) in its i-th component
        - q: vector containing Q(i) in its i-th component

    Return:
        - js: JS-divergence between p and q    
    """
    m = 0.5 * (p + q)
    js = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

    return js


def batch_divergence(batch_p, batch_q, n, div_fn):
    """
    Computes divergence per sample between two batches containing
    probability distributions in each row. The divergence is computed
    between the i-th row of the first batch and the i-th row of the
    second batch for every i.

    Args:
        - batch_p: tensor of size [batch_size, n_classes]
        - batch_q: tensor of size [batch_size, n_classes]
        - n: the number of probabilities per row
        - div_fn: divergence function to be used
    """
    all_probs = tf.concat([batch_p, batch_q], axis=1)
    fn = (lambda x: div_fn(x[:n], x[n:]))
    divergences = tf.map_fn(fn, all_probs)

    return divergences


def cosine_similarities(A, B):
    """
    Args:
        - A: tensor containing samples row-wise
        - B: tensor containing samples row-wise

    Return:
        - similarities: tensor containing cosine
          similarities of paired samples
    """
    dot_prods = tf.reduce_sum(A * B, axis=1)

    A_norms = tf.sqrt(tf.reduce_sum(A * A, axis=1))
    B_norms = tf.sqrt(tf.reduce_sum(B * B, axis=1))

    similarities = (dot_prods / A_norms) / B_norms

    return tf.abs(similarities)


# Reliability measures for ICC computation
def get_batch_dimensions(X):
    d = tf.cast(tf.shape(X)[1], tf.float32)
    n = tf.cast(tf.shape(X)[0], tf.float32)

    return n, d


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


def per_feature_batch_ICC(test_batch, retest_batch, icc_func):
    """
    Args:

    Return:
        iccs: tensor containg the ICC value for each feature
    """
    n_features = test_batch.get_shape()[1]
    iccs = []

    for i in range(n_features):
        Y_1 = tf.reshape(test_batch[:, i], [-1, 1])
        Y_2 = tf.reshape(retest_batch[:, i], [-1, 1])
        Y = tf.concat([Y_1, Y_2], axis=1)
        icc = icc_func(Y)
        iccs.append(icc)

    iccs = tf.stack(iccs)
    return iccs
