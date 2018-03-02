import tensorflow as tf
from base import NetworkHeadBase
from src.features import all_features


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
        self.labels_ids = hash_table.lookup(features[predict])
        self.labels = tf.reshape(tf.one_hot(
                self.labels_ids,
                depth=num_buckets,
            ),
            [-1, num_buckets],
        )
        self.predictions = model.gen_head(
            last_layer,
            num_buckets,
        )
        self.loss = tf.losses.softmax_cross_entropy(
            self.labels,
            self.predictions,
        )
        self.predict = predict
        # TODO: Handling of ties!!
        self.metrics = {
            'in_top_%s' % k: tf.reduce_mean(tf.cast(tf.nn.in_top_k(
                predictions=self.predictions,
                targets=tf.reshape(self.labels_ids, [-1]),
                k=k,
            ), tf.float32))
            for k in [1, 5, 10, 20, 50]
        }
        super(CategoricalVariableClassificationHead, self).__init__(
            name=name,
            model=model,
            last_layer=last_layer,
            features=features,
            **kwargs
        )

    def get_tags(self):
        tags = super(CategoricalVariableClassificationHead, self).get_tags()
        feature_info = all_features.feature_info
        tags.append('predict_%s' % feature_info[self.predict]['shortname'])
        return tags
