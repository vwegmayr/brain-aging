import copy
import tensorflow as tf
from base import NetworkHeadBase
from src.features import all_features


class ClassificationHead(NetworkHeadBase):
    def __init__(
        self,
        # NetworkHeadBase arguments
        name,
        model,
        last_layer,
        features,
        # Custom arguments (from config file)
        predict,
        # Args passed to parent
        **kwargs
    ):
        """
        Arguments:
        - predict:
          List of boolean features to predict
          Example: ['is_class_a', 'is_class_b']
        """
        self.predict_feature_names = predict
        self.predictions = model.gen_head(
            last_layer,
            len(predict),
        )

        # Compute loss
        self.labels = [features[ft_name] for ft_name in predict]
        self.labels = tf.concat(self.labels, 1)
        self.loss = tf.losses.softmax_cross_entropy(
            self.labels,
            self.predictions,
        )

        # Metrics for training/eval
        batch_size = tf.shape(self.labels)[0]
        accuracy = tf.reduce_mean(tf.cast(
            tf.equal(
                tf.argmax(self.predictions, 1),
                tf.argmax(self.labels, 1),
            ),
            tf.float32,
        ))
        self.metrics = {
            'accuracy': accuracy
        }
        self.metrics.update({
            'predicted_%s_ratio' % ft_name:
            tf.reduce_sum(tf.cast(
                tf.equal(tf.argmax(self.predictions, 1), i),
                tf.float32,
            )) / tf.cast(batch_size, tf.float32)
            for i, ft_name in enumerate(self.predict_feature_names)
        })

        super(ClassificationHead, self).__init__(
            name=name,
            model=model,
            last_layer=last_layer,
            features=features,
            **kwargs
        )

    def get_evaluated_metrics(self):
        evaluation_metrics = \
            super(ClassificationHead, self).get_evaluated_metrics()

        predicted = tf.argmax(self.predictions, 1)
        actual_class = tf.argmax(self.labels, 1)
        evaluation_metrics.update({
            'false_negatives': tf.metrics.false_negatives(
                predicted, actual_class),
            'false_positives': tf.metrics.false_positives(
                predicted, actual_class),
        })
        return evaluation_metrics

    def get_predictions(self):
        predictions = super(ClassificationHead, self).get_predictions()
        predictions.update({
            ft_name: self.predictions[:, i]
            for i, ft_name in enumerate(self.predict_feature_names)
        })
        return predictions

    def get_tags(self):
        tags = super(ClassificationHead, self).get_tags()
        tags.append('classification')
        feature_info = all_features.feature_info
        ft_names = [
            feature_info[n]['shortname']
            for n in self.predict_feature_names
        ]
        ft_names.sort()
        tags.append('_'.join(ft_names))
        return tags
