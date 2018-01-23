import tensorflow as tf
from modules.models.base import BaseTF as TensorflowBaseEstimator
from modules.models.utils import parse_hooks

from preprocessing import preprocess_all_if_needed
from input import train_input, test_input
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

        prediction_info = params['predicted']
        num_classes = len(prediction_info)
        regression = num_classes == 1
        predicted_features = [i['feature'] for i in prediction_info]
        predicted_features_avg = [i['average'] for i in prediction_info]

        m = Model(is_training=(mode == tf.estimator.ModeKeys.TRAIN))
        last_layer = m.gen_last_layer(features)
        if regression:
            predictions = m.gen_head_regressor(
                last_layer,
                predicted_features_avg,
            )
            compute_loss_fn = tf.losses.mean_squared_error
        else:
            predictions = m.gen_head_classifier(
                last_layer,
                num_classes,
            )
            compute_loss_fn = tf.losses.softmax_cross_entropy

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={
                    ft_name: predictions[:, i]
                    for i, ft_name in enumerate(predicted_features)
                },
                export_outputs={
                    'outputs': tf.estimator.export.PredictOutput({
                        ft_name: predictions[:, i]
                        for i, ft_name in enumerate(predicted_features)
                    })
                }
            )

        labels = [features[ft_name] for ft_name in predicted_features]
        labels = tf.concat(labels, 1)

        if regression:
            eval_metric_ops = {
                'rmse': tf.metrics.root_mean_squared_error(
                    tf.cast(labels, tf.float32),
                    predictions,
                ),
                'rmse_vs_avg': tf.metrics.root_mean_squared_error(
                    tf.cast(labels, tf.float32),
                    predictions*0.0 + predicted_features_avg,
                ),
            }
        else:
            eval_metric_ops = {
                'accuracy': tf.metrics.accuracy(
                    tf.argmax(predictions, 1),
                    tf.argmax(labels, 1)
                ),
                'false_negatives': tf.metrics.false_negatives(
                    tf.argmax(predictions, 1),
                    tf.argmax(labels, 1)
                ),
                'false_positives': tf.metrics.false_positives(
                    tf.argmax(predictions, 1),
                    tf.argmax(labels, 1)
                ),
            }

        loss = compute_loss_fn(labels, predictions)
        loss_v_avg = compute_loss_fn(
            tf.cast(labels, tf.float32),
            tf.cast(labels, tf.float32)*0.0 + predicted_features_avg,
        )
        accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1)),
            tf.float32,
        ))

        log_variables = {
            "loss": loss,
            "loss_v_avg": loss_v_avg,
            "accuracy": accuracy,
        }
        if not regression:
            log_variables.update({
                'count_predicted_%s' % predicted_features[i]:
                tf.reduce_sum(tf.cast(
                    tf.equal(tf.argmax(predictions, 1), i),
                    tf.float32,
                ))
                for i in range(num_classes)
            })

        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step(),
        )

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops,
            training_hooks=self.get_training_hooks(
                params,
                log_variables=log_variables,
            ),
        )

    def get_training_hooks(self, params, log_variables):
        if "hooks" in params:
            training_hooks = parse_hooks(
                params["hooks"],
                locals(),
                self.save_path)
        else:
            training_hooks = []

        if "log_loss_every_n_iter" in params:
            hook_logged = log_variables.copy()
            hook_logged.update({
                "global_step": tf.train.get_global_step(),
            })
            training_hooks.append(
                tf.train.LoggingTensorHook(
                    hook_logged,
                    every_n_iter=params["log_loss_every_n_iter"],
                )
            )
        return training_hooks

    def compute_loss(self, labels, predictions):
        return tf.losses.mean_squared_error(labels, predictions)

    def gen_input_fn(self, X, y=None, train=True, input_fn_config={}):
        # TODO: Return "test_input" for testing
        preprocess_all_if_needed(input_fn_config['data_generation'])

        def _input_fn():
            if train:
                return train_input(
                    input_fn_config['data_generation'],
                    input_fn_config['data_streaming'],
                )
            else:
                return test_input(
                    input_fn_config['data_generation'],
                    input_fn_config['data_streaming'],
                )
        return _input_fn

    def validate_params(self, params):
        assert("regression" in params ^ "classification" in params)
