import tensorflow as tf
from base import NetworkHeadBase
from src.features import all_features


class RegressionHead(NetworkHeadBase):
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
          List of features to predict, with a guess of their average
          Example: [{'feature': 'age', 'average': 50}]
        """
        self.predict_feature_names = [i['feature'] for i in predict]
        self.predict_feature_avg = [i['average'] for i in predict]
        self.predictions = model.gen_head(
            last_layer,
            len(predict),
            nl=tf.identity,
        ) + self.predict_feature_avg

        # Compute loss
        self.labels = [features[p['feature']] for p in predict]
        self.labels = tf.concat(self.labels, 1)

        self.loss = tf.losses.mean_squared_error(self.labels, self.predictions)
        self.loss_v_avg = tf.losses.mean_squared_error(
            self.labels,
            tf.zeros_like(self.labels, tf.float32) + self.predict_feature_avg,
        )

        super(RegressionHead, self).__init__(
            name=name,
            model=model,
            last_layer=last_layer,
            features=features,
            **kwargs
        )

    def get_logged_training_variables(self):
        training_variables = \
            super(RegressionHead, self).get_logged_training_variables()
        training_variables.update({
            'loss_v_avg': self.loss_v_avg
        })
        return training_variables

    def get_evaluated_metrics(self):
        evaluation_metrics = \
            super(RegressionHead, self).get_evaluated_metrics()
        evaluation_metrics.update({
            'rmse': tf.metrics.root_mean_squared_error(
                tf.cast(self.labels, tf.float32),
                self.predictions,
            ),
            'rmse_vs_avg': tf.metrics.root_mean_squared_error(
                tf.cast(self.labels, tf.float32),
                tf.zeros_like(
                    self.labels,
                    tf.float32,
                ) + self.predict_feature_avg,
            ),
        })
        return evaluation_metrics

    def get_predictions(self):
        predictions = super(RegressionHead, self).get_predictions()
        predictions.update({
            ft_name: self.predictions[:, i]
            for i, ft_name in enumerate(self.predict_feature_names)
        })
        return predictions

    def get_tags(self):
        tags = super(RegressionHead, self).get_tags()
        tags.append('regression')
        feature_info = all_features.feature_info
        for n in sorted(self.predict_feature_names):
            tags.append('reg_%s' % feature_info[n]['shortname'])
        return tags
