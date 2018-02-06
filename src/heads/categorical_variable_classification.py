import tensorflow as tf
from base import NetworkHeadBase


class CategoricalVariableClassificationHead(NetworkHeadBase):
    def __init__(
        self,
        # NetworkHeadBase arguments
        name,
        model,
        last_layer,
        features,
        # Custom arguments (from config file)
        predict,
        num_buckets,
        # Args passed to parent
        **kwargs
    ):
        hash_table = tf.contrib.lookup.IdTableWithHashBuckets(
            None,
            num_buckets,
            key_dtype=tf.int64,
        )
        labels = hash_table.lookup(features[predict])
        labels = tf.reshape(tf.one_hot(
                labels,
                depth=num_buckets,
            ),
            [-1, num_buckets],
        )
        predictions = model.gen_head(
            last_layer,
            num_buckets,
        )
        self.loss = tf.losses.softmax_cross_entropy(
            labels,
            predictions,
        )
        super(CategoricalVariableClassificationHead, self).__init__(
            name=name,
            model=model,
            last_layer=last_layer,
            features=features,
            **kwargs
        )
