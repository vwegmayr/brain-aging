import tensorflow as tf
from modules.models.base import BaseTF as TensorflowBaseEstimator
from modules.models.utils import parse_hooks

from preprocessing import preprocess_all_if_needed
from input import train_input
from model import Model


class Estimator(TensorflowBaseEstimator):
    """docstring for Estimator"""

    def __init__(self, *args, **kwargs):
        import features
        super(Estimator, self).__init__(*args, **kwargs)
        features.all_features.feature_info[features.MRI]['shape'] = \
            self.input_fn_config['data_generation']['image_shape']
        self.feature_spec = {
            name: tf.placeholder(
                    shape=[None] + ft_info['shape'],
                    dtype=ft_info['type']
                )
            for name, ft_info in features.all_features.feature_info.items()
        }

    def score(self, X, y):
        """
        Only used for prediction apparently. Dont need it now.
        """
        assert(False)

    def model_fn(self, features, labels, mode, params, config):
        """
        https://www.tensorflow.org/extend/estimators#constructing_the_model_fn
        - features: features returned by @gen_input_fn
        - labels: None (not used)
        - mode: {train, evaluate, inference}
        - params: parameters from yaml config file
        - config: tensorflow.python.estimator.run_config.RunConfig
        """

        predicted_feature = params['predicted_feature']
        predicted_feature_avg = params['predicted_feature_avg']
        labels = features[predicted_feature]
        m = Model(is_training=(mode == tf.estimator.ModeKeys.TRAIN))
        predictions = m.gen_output(features)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={predicted_feature: predictions},
                export_outputs={
                    'outputs': tf.estimator.export.PredictOutput({
                        predicted_feature: predictions
                    })
                }
            )

        loss = tf.losses.mean_squared_error(labels, predictions)
        loss_v_avg = tf.losses.mean_squared_error(
            tf.cast(labels, tf.float32),
            tf.cast(labels, tf.float32)*0.0 + predicted_feature_avg,
        )

        # Calculate root mean squared error as additional eval metric
        eval_metric_ops = {
            'rmse': tf.metrics.root_mean_squared_error(
                tf.cast(labels, tf.float32),
                predictions,
            ),
            'rmse_vs_avg': tf.metrics.root_mean_squared_error(
                tf.cast(labels, tf.float32),
                predictions*0.0 + predicted_feature_avg,
            ),
        }

        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )

        if "hooks" in params:
            training_hooks = parse_hooks(
                params["hooks"],
                locals(),
                self.save_path)
        else:
            training_hooks = []

        if "log_loss_every_n_iter" in params:
            training_hooks.append(
                tf.train.LoggingTensorHook({
                        "loss": loss,
                        "loss_v_avg": loss_v_avg,
                    },
                    every_n_iter=params["log_loss_every_n_iter"],
                )
            )

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            training_hooks=training_hooks)

    def gen_input_fn(self, X, y=None, input_fn_config={}):
        # TODO: Return "test_input" for testing
        preprocess_all_if_needed(input_fn_config['data_generation'])

        def _train_input():
            return train_input(
                input_fn_config['data_generation'],
                input_fn_config['data_streaming'],
            )
        return _train_input
