import tensorflow as tf
import numpy as np
import pickle as pkl
import json
import copy
import sys

from modules.models.base import BaseTF as TensorflowBaseEstimator
from modules.models.utils import custom_print
import features as ft_def
import src.data.providers
from model import Model
from train_hooks import PrintAndLogTensorHook


class Estimator(TensorflowBaseEstimator):
    """docstring for Estimator"""

    def __init__(self, run_config, sumatra_outcome_config, *args, **kwargs):
        self.is_model_first_run = True
        self.run_config = run_config
        self.sumatra_outcome_config = sumatra_outcome_config
        self.sumatra_outcome = {}
        self.sumatra_outcome['run_tags'] = []

        tf_run_config = copy.deepcopy(run_config['tf_estimator_run_config'])
        if 'session_config' in tf_run_config:
            session_config = tf_run_config['session_config']
            gpu_config = session_config.get('gpu_options')
            if gpu_config is not None:
                del session_config['gpu_options']
            session_config_obj = tf.ConfigProto(**session_config)
            if gpu_config is not None:
                for conf_key, conf_val in gpu_config.items():
                    setattr(session_config_obj.gpu_options, conf_key, conf_val)
            tf_run_config['session_config'] = session_config_obj

        super(Estimator, self).__init__(
            config=tf_run_config,
            *args,
            **kwargs
        )

        # Data provider
        provider = self.input_fn_config['data_provider']
        module = getattr(src.data.providers, provider)
        if module is None:
            custom_print(
                'FATAL: Data provider ' +
                self.input_fn_config['data_provider'] +
                ' not found'
            )
        assert(module is not None)
        self.data_provider = module.DataProvider(self.input_fn_config)
        ft_def.all_features.feature_info[ft_def.MRI]['shape'] = \
            self.data_provider.get_mri_shape()

        self.feature_spec = {
            name: tf.placeholder(
                    shape=[1] + ft_info['shape'],
                    dtype=ft_info['type']
                )
            for name, ft_info in ft_def.all_features.feature_info.items()
            if not ft_info['only_for_extractor']
        }

    def fit_main_training_loop(self, X, y):
        """
        Trains and runs validation regularly at the same time
        """
        self.evaluations = []
        self.training_metrics = []

        custom_print('[INFO] Main training loop. Model dir is %s' % (
            self.config["model_dir"]))
        if 'reason' in self.sumatra_outcome_config:
            custom_print('[INFO] Provided run reason in config: %s' %
                self.sumatra_outcome_config['reason'])

        printed_count = [0]  # Workaround to modify variable inside nested func

        def one_run_finished(t='.'):
            sys.stdout.write(t)
            printed_count[0] += 1
            if printed_count[0] % 10 == 0:
                sys.stdout.write('\n')
            sys.stdout.flush()
            self.write_outcome()

        def do_evaluate():
            evaluate_fn = self.gen_input_fn(X, y, False, self.input_fn_config)
            assert(evaluate_fn is not None)
            self.evaluations.append(
                    self.estimator.evaluate(input_fn=evaluate_fn)
                )
            one_run_finished('V')

        num_epochs = self.run_config['num_epochs']
        validations_per_epoch = self.run_config['validations_per_epoch']

        one_run_finished('\n')
        # 1st case, evaluation every few epochs
        if validations_per_epoch <= 1:
            validation_counter = 0
            train_fn = self.gen_input_fn(X, y, True, self.input_fn_config)
            for i in range(num_epochs):
                self.estimator.train(input_fn=train_fn)
                one_run_finished()

                # Check if we need to run validation
                validation_counter += validations_per_epoch
                if validation_counter >= 1:
                    validation_counter -= 1
                    do_evaluate()

        # 2nd case, several evaluations per epoch (should be an int then!)
        else:
            iters = num_epochs * validations_per_epoch
            for i in range(iters):
                train_fn = self.gen_input_fn(
                    X, y, True, self.input_fn_config,
                    shard=(i % validations_per_epoch, validations_per_epoch),
                )
                self.estimator.train(input_fn=train_fn)
                one_run_finished()
                do_evaluate()

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
        NETWORK_BODY_SCOPE = 'network_body'
        network_heads = params['network_heads']

        if mode == tf.estimator.ModeKeys.PREDICT:
            features = self.data_provider.predict_features(features)

        with tf.variable_scope(NETWORK_BODY_SCOPE):
            m = Model(
                is_training=(mode == tf.estimator.ModeKeys.TRAIN),
                print_shapes=self.is_model_first_run,
            )
            self.is_model_first_run = False
            last_layer = m.gen_last_layer(features, **params['network_body'])

        heads = []
        for head_name, _h in network_heads.items():
            h = copy.deepcopy(_h)
            _class = h['class']
            del h['class']
            with tf.variable_scope(head_name):
                head = _class(
                    name=head_name,
                    model=m,
                    last_layer=last_layer,
                    features=features,
                    **h
                )
                self.sumatra_add_tags(head.get_tags())
                heads.append(head)

        # TODO: Not sure what I'm doing here
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {}
            for head in heads:
                with tf.variable_scope(head.get_name()):
                    predictions.update(head.get_predictions())
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={
                    ft_name: ft_val
                    for ft_name, ft_val in predictions.items()
                },
                export_outputs={
                    'outputs': tf.estimator.export.PredictOutput({
                        ft_name: ft_val
                        for ft_name, ft_val in predictions.items()
                    })
                }
            )

        # Compute loss
        global_loss = 0
        for head in heads:
            global_loss += head.get_global_loss_contribution()

        # Variables logged during training (append head name as prefix)
        train_log_variables = {
            "global_optimizer_loss": global_loss,
        }
        for head in heads:
            with tf.variable_scope(head.get_name()):
                variables = head.get_logged_training_variables()
                train_log_variables.update({
                    head.name + '/' + var_name: var_value
                    for var_name, var_value in variables.items()
                })

        # Metrics for evaluation
        eval_metric_ops = {}
        for head in heads:
            with tf.variable_scope(head.get_name()):
                variables = head.get_evaluated_metrics()
                eval_metric_ops.update({
                    head.name + '/' + var_name: var_value
                    for var_name, var_value in variables.items()
                })

        # Optimizer
        train_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=NETWORK_BODY_SCOPE,
        )
        for head in heads:
            head.register_globally_trained_variables(train_vars)

        train_ops = self.generate_train_ops(
            train_log_variables,
            global_loss,
            train_vars,
            heads,
            **params['network_train_ops_settings']
        )

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=global_loss,
            train_op=tf.group(*train_ops),
            eval_metric_ops=eval_metric_ops,
            training_hooks=self.get_training_hooks(
                params,
                log_variables=train_log_variables,
            ),
        )

    def generate_train_ops(
        self,
        train_log_variables,
        global_loss,
        train_vars,
        heads,
        # Arguments from Config
        alternative_training_steps=0,
        adam_aggregation_method='DEFAULT',
    ):
        """
        Generates training operations for the network.
        Basically it is `global_train_op` and `heads_train_op`, but it's
        possible to train alternatively (with `alternative_training_steps` > 0)
        """
        adam_aggregation_method = getattr(
            tf.AggregationMethod,
            adam_aggregation_method,
        )
        optimizer = tf.train.AdamOptimizer(0.00005)

        def global_train_op():
            return optimizer.minimize(
                loss=global_loss,
                var_list=train_vars,
                aggregation_method=adam_aggregation_method,
            )

        def heads_train_op():
            ops = []
            for head in heads:
                with tf.variable_scope(head.get_name()):
                    ops.append(head.get_head_train_op(
                        optimizer,
                        aggregation_method=adam_aggregation_method,
                    ))
            return tf.group(*ops)

        global_step = tf.train.get_global_step()
        global_step_incr = tf.assign(global_step, global_step+1)

        if alternative_training_steps == 0:
            return [global_step_incr, global_train_op(), heads_train_op()]

        # Rounds based training
        _round = tf.cast(
            tf.mod(tf.floordiv(global_step, alternative_training_steps), 2),
            tf.int32,
        )
        train_log_variables['_round'] = _round

        return [
            global_step_incr,
            tf.cond(
                tf.equal(_round, 0),
                global_train_op,
                lambda: tf.no_op(),
                name="condRoundEq0",
            ),
            tf.cond(
                tf.equal(_round, 1),
                heads_train_op,
                lambda: tf.no_op(),
                name="condRoundEq1",
            ),
        ]

    def get_training_hooks(self, params, log_variables):
        training_hooks = []

        if "train_log_every_n_iter" in params:
            hook_logged = log_variables.copy()
            hook_logged.update({
                "global_step": tf.train.get_global_step(),
            })
            training_hooks.append(
                PrintAndLogTensorHook(
                    self,
                    tensors=hook_logged,
                    every_n_iter=params["train_log_every_n_iter"],
                )
            )
        if "hooks" in params:
            for h in params["hooks"]:
                args = copy.copy(h["params"])
                if "ckptdir" in args:
                    args["ckptdir"] = self.save_path
                training_hooks.append(h["class"](**args))
        return training_hooks

    def compute_loss(self, labels, predictions):
        return tf.losses.mean_squared_error(labels, predictions)

    def gen_input_fn(
        self,
        X,
        y=None,
        train=True,
        input_fn_config={},
        shard=None,
    ):
        return self.data_provider.get_input_fn(train, shard)

    def training_log_values(self, values):
        self.training_metrics.append(values)

    def write_outcome(self):
        """
        Creates sumatra output file where the values of evaluation/train
        metrics are reported. Also provides sumatra run reason/outcome/tags.
        """
        validations_per_epoch = self.run_config['validations_per_epoch']
        output_dir = self.config["model_dir"]

        self.generate_numeric_outcome()

        config_sumatra = self.sumatra_outcome_config
        if ('reason' in config_sumatra and
                config_sumatra['reason'] is not None):
            self.sumatra_outcome['run_reason'] = config_sumatra['reason']
        if ('tags' in config_sumatra and
                config_sumatra['tags'] is not None):
            self.sumatra_add_tags(config_sumatra['tags'])

        with open('%s/eval_values.pkl' % (output_dir), 'wb') as f:
            pkl.dump({
                'version': 1,
                'validations_per_epoch': validations_per_epoch,
                'evaluate': self.evaluations,
                'train': self.training_metrics,
            }, f, pkl.HIGHEST_PROTOCOL)

        # Backward compatibility for sumatra format and metric names
        sumatra_metrics = self.sumatra_outcome['numeric_outcome']
        old_new_stats = self.run_config['stats_backward_compatibility']
        for new_name, old_name in old_new_stats.items():
            if new_name in sumatra_metrics and old_name not in sumatra_metrics:
                cpy = copy.deepcopy(sumatra_metrics[new_name])
                cpy['_deprecated'] = True
                sumatra_metrics[old_name] = cpy

        with open('%s/sumatra_outcome.json' % (output_dir), 'w') as outfile:
            json.dump(self.sumatra_outcome, outfile)

    def sumatra_add_tags(self, tags_list):
        self.sumatra_outcome['run_tags'] = list(set(
            self.sumatra_outcome['run_tags'] + tags_list
        ))

    def generate_numeric_outcome(self):
        sumatra_metrics = self.sumatra_outcome['numeric_outcome'] = {}
        if self.evaluations == []:
            return
        validations_per_epoch = self.run_config['validations_per_epoch']
        last_epoch = len(self.evaluations) / float(validations_per_epoch)

        # List of dicts to dict of lists
        v_eval = dict(zip(
            self.evaluations[0],
            zip(*[d.values() for d in self.evaluations])
        ))
        v_train = dict(zip(
            self.training_metrics[0],
            zip(*[d.values() for d in self.training_metrics])
        ))

        for v, prefix in [[v_eval, 'eval/'], [v_train, 'train/']]:
            for label, values in v.items():
                # Need to skip first value, because loss is not evaluated
                # at the beginning
                x_values = np.linspace(
                    0,
                    last_epoch,
                    len(values)+1,
                )

                # All this data needs to be serializable, so get rid of
                # numpy arrays, np.float32 etc..
                sumatra_metrics[prefix + label] = {
                    'type': 'numeric',
                    'x': x_values[1:].tolist(),
                    'x_label': 'Training epoch',
                    'y': np.array(values).tolist(),
                }

        accuracy_key = 'classifier/mean_per_class_accuracy'
        if accuracy_key in v_eval and len(v_eval[accuracy_key]) >= 8:
            accuracy = v_eval[accuracy_key]
            last_n = int(len(accuracy)*0.25)
            accuracy = accuracy[len(accuracy)-last_n:]
            self.sumatra_outcome['text_outcome'] = \
                'Per class accuracy %s (std=%.3f on the last %s steps)' % (
                    np.mean(accuracy),
                    np.std(accuracy),
                    last_n,
                )
        else:
            self.sumatra_outcome['text_outcome'] = 'TODO'
