import copy
import tensorflow as tf
import numpy as np
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
        loss_classes_weights={},
        num_classes_in_evaluation=None,
        # Args passed to parent
        **kwargs
    ):
        """
        Arguments:
        - predict:
          List of boolean features to predict
          Example: ['is_class_a', 'is_class_b']
        """
        if num_classes_in_evaluation is None:
            num_classes_in_evaluation = len(predict)
        assert(num_classes_in_evaluation <= len(predict))
        self.num_classes_in_evaluation = num_classes_in_evaluation
        self.predict_feature_names = predict
        self.predictions = model.gen_head(
            last_layer,
            len(predict),
        )

        # Compute loss
        weights_per_class = {ft_name: 1.0 for ft_name in predict}
        weights_per_class.update(loss_classes_weights)
        weights_per_class = np.array([weights_per_class[ft_name] for ft_name in predict])
        self.labels = [features[ft_name] for ft_name in predict]
        self.labels = tf.concat(self.labels, 1)
        self.loss = tf.losses.softmax_cross_entropy(
            self.labels,
            self.predictions,
            reduction=tf.losses.Reduction.MEAN,
            weights=tf.reduce_sum(
                tf.cast(self.labels, tf.float32) * weights_per_class, 1,
            ),
        )

        # Metrics for training/eval
        batch_size = tf.shape(self.labels)[0]
        accuracy = tf.cast(
            tf.equal(
                tf.argmax(self.predictions, 1),
                tf.argmax(self.labels, 1),
            ),
            tf.float32,
        )
        self.metrics = {
            'accuracy': tf.reduce_mean(accuracy),
        }
        self.metrics.update({
            'predicted_%s_ratio' % ft_name:
            tf.reduce_sum(tf.cast(
                tf.equal(tf.argmax(self.predictions, 1), i),
                tf.float32,
            )) / tf.cast(batch_size, tf.float32)
            for i, ft_name in enumerate(self.predict_feature_names)
        })

        # Accuracy on different classes
        def accuracy_on_class(i):
            weights = tf.cast(
                tf.equal(
                    tf.argmax(self.labels, 1),
                    i,
                ),
                tf.float32,
            )
            class_acc = tf.reduce_sum(weights * accuracy)
            class_acc /= tf.reduce_sum(weights) + 0.0001
            return class_acc, tf.reduce_sum(weights)
        self.classes_accuracy = {
            'accuracy_on_%s' % c: accuracy_on_class(i)
            for i, c in enumerate(predict)
        }
        self.metrics.update({
            name: v[0]
            for name, v in self.classes_accuracy.items()
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

        predicted = tf.argmax(
            self.predictions[:, 0:self.num_classes_in_evaluation],
            1,
        )
        actual_class = tf.argmax(self.labels, 1)
        evaluation_metrics.update({
            'false_negatives': tf.metrics.false_negatives(
                actual_class, predicted),
            'false_positives': tf.metrics.false_positives(
                actual_class, predicted),
            'mean_per_class_accuracy': tf.metrics.mean_per_class_accuracy(
                actual_class, predicted, self.num_classes_in_evaluation),
        })
        evaluation_metrics.update({
            n: tf.metrics.mean(v[0], weights=v[1], name='%s_weighted' % n)
            for n, v in self.classes_accuracy.items()
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
