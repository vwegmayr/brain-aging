import tensorflow as tf
from base import NetworkHeadBase


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
        self.predicted_features_names = [i['feature'] for i in predict]
        self.predicted_features_avg = [i['average'] for i in predict]
        self.predictions = model.gen_head(
            last_layer,
            len(predict),
            nl=tf.identity,
        ) + self.predicted_features_avg

        # Compute loss
        self.labels = [features[p['feature']] for p in predict]
        self.labels = tf.concat(self.labels, 1)

        self.loss = tf.losses.mean_squared_error(self.labels, self.predictions)
        self.loss_v_avg = tf.losses.mean_squared_error(
            self.labels,
            self.labels*0.0 + self.predicted_features_avg,
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
                self.labels,
                self.predictions,
            ),
            'rmse_vs_avg': tf.metrics.root_mean_squared_error(
                self.labels,
                self.predictions*0.0 + self.predicted_features_avg,
            ),
        })
        return evaluation_metrics

    def get_predictions(self):
        predictions = super(RegressionHead, self).get_predictions()
        predictions.update({
            ft_name: self.predictions[:, i]
            for i, ft_name in enumerate(self.predicted_features_names)
        })
        return predictions
