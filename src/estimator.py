import tensorflow as tf
import numpy as np
import pickle as pkl
import json

from modules.models.base import BaseTF as TensorflowBaseEstimator
from modules.models.utils import parse_hooks, custom_print
from preprocessing import preprocess_all_if_needed
from input import train_input, test_input
from model import Model


class Estimator(TensorflowBaseEstimator):
    """docstring for Estimator"""

    def __init__(self, run_config, *args, **kwargs):
        import features
        self.run_config = run_config
        self.sumatra_outcome = {}
        super(Estimator, self).__init__(
            config=run_config['tf_run_config'],
            *args,
            **kwargs
        )
        features.all_features.feature_info[features.MRI]['shape'] = \
            self.input_fn_config['data_generation']['image_shape']
        self.feature_spec = {
            name: tf.placeholder(
                    shape=[None] + ft_info['shape'],
                    dtype=ft_info['type']
                )
            for name, ft_info in features.all_features.feature_info.items()
        }

    def fit_main_training_loop(self, X, y):
        """
        Trains and runs validation regularly at the same time
        """
        train_fn = self.gen_input_fn(X, y, True, self.input_fn_config)
        evaluate_fn = self.gen_input_fn(X, y, False, self.input_fn_config)
        num_epochs = self.run_config['num_epochs']
        validations_per_epoch = self.run_config['validations_per_epoch']
        assert(evaluate_fn is not None)
        # TODO: Support more validations per epoch
        assert(validations_per_epoch <= 1)

        validation_counter = 0
        self.evaluations = []
        for i in range(num_epochs):
            self.estimator.train(input_fn=train_fn)

            # Check if we need to run validation
            validation_counter += validations_per_epoch
            if validation_counter >= 1:
                validation_counter -= 1
                self.evaluations.append(
                    self.estimator.evaluate(input_fn=evaluate_fn)
                )
                self.export_evaluation_stats()


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

    def export_evaluation_stats(self):
        """
        @values is a list of return values of tf.Estimator.evaluate
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if self.evaluations == []:
            return
        output_dir = self.config["model_dir"]
        validations_per_epoch = self.run_config['validations_per_epoch']
        custom_print('[INFO] Exporting evaluations to "%s"' % output_dir)

        # List of dicts to dict of lists
        v = dict(zip(
            self.evaluations[0],
            zip(*[d.values() for d in self.evaluations])
        ))

        self.sumatra_outcome['numeric_outcome'] = {}

        for label, values in v.items():
            # Need to skip first value, because loss is not evaluated
            # at the beginning
            x_values = np.linspace(
                0,
                len(values) / validations_per_epoch,
                len(values)+1,
            )
            plt.plot(
                x_values[1:],
                values,
            )
            plt.title(label)
            plt.xlabel('Training iteration')
            plt.ylabel('%s on validation set' % label)
            plt.savefig(
                '%s/eval_%s.png' % (output_dir, label),
                bbox_inches='tight',
            )
            plt.close()

            # All this data needs to be serializable, so get rid of
            # numpy arrays, np.float32 etc..
            self.sumatra_outcome['numeric_outcome'][label] = {
                'type': 'numeric',
                'x': x_values[1:].tolist(),
                'y': np.array(values).tolist(),
            }

        if 'accuracy' in v and len(v['accuracy']) > 3:
            self.sumatra_outcome['text_outcome'] = \
                'Final accuracy %s' % (v['accuracy'][-1])
        else:
            self.sumatra_outcome['text_outcome'] = 'TODO'

        with open('%s/eval_values.pkl' % (output_dir), 'wb') as f:
            pkl.dump({
                'version': 1,
                'validations_per_epoch': validations_per_epoch,
                'evaluate': self.evaluations,
            }, f, pkl.HIGHEST_PROTOCOL)

        with open('%s/sumatra_outcome.json' % (output_dir), 'w') as outfile:
            json.dump(self.sumatra_outcome, outfile)
