import tensorflow as tf
import numpy as np


class PrintAndLogTensorHook(tf.train.LoggingTensorHook):
    def __init__(
        self,
        estimator,
        print_summary_init_value=0.5,
        print_summary_tensor=None,
        **kwargs
    ):
        super(PrintAndLogTensorHook, self).__init__(**kwargs)
        self.estimator = estimator
        self.print_summary_tensor = print_summary_tensor
        self.print_aggregated_history = [print_summary_init_value] * 30

    def _log_tensors(self, tensor_values):
        """
        tensor_values is a dict {string => tensor_value }
        """
        self.estimator.training_log_values(tensor_values)
        # super(PrintAndLogTensorHook, self)._log_tensors(tensor_values)
        if self.print_summary_tensor is not None:
            self.print_aggregated_history.append(
                tensor_values[self.print_summary_tensor]
            )
            print(
                'Entrack',
                'Step:', tensor_values['global_step'],
                'Loss:', tensor_values['global_optimizer_loss'],
                self.print_summary_tensor, np.mean(
                    self.print_aggregated_history[-20:-1]
                )
            )


class SessionHookFullTrace(tf.train.SessionRunHook):
    """Hook to perform Traces every N steps."""

    def __init__(
        self,
        ckptdir,
        first_step=True,
        every_step=50,
        trace_level=tf.RunOptions.FULL_TRACE,
    ):
        self.writer = tf.summary.FileWriter(ckptdir)
        self.trace_level = trace_level
        self.every_step = every_step
        self.first_step = first_step
        self._trace = False

    def begin(self):
        self._global_step_tensor = tf.train.get_global_step()
        if self._global_step_tensor is None:
            raise RuntimeError("Global step should be created to use SessionHookFullTrace.")

    def before_run(self, run_context):
        if self._trace:
            options = tf.RunOptions(trace_level=self.trace_level)
        else:
            options = None
        return tf.train.SessionRunArgs(fetches=self._global_step_tensor,
                                       options=options)

    def after_run(self, run_context, run_values):
        global_step = run_values.results - 1
        if self._trace:
            self._trace = False
            self.writer.add_run_metadata(
                run_values.run_metadata,
                '{global_step}'.format(global_step=global_step),
                global_step,
            )
        if (self.every_step is not None and
            not (global_step + 1) % self.every_step):
            self._trace = True
        if self.first_step and global_step == 1:
            self._trace = True
