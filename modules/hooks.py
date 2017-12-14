import numpy as np
import datetime

from tensorflow.python.training.session_run_hook import SessionRunHook
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer
from tensorflow.python.platform import tf_logging as logging


def humanize_time(secs):
    mins, secs = divmod(secs, 60)
    hours, mins = divmod(mins, 60)
    return '%02d:%02d:%02d' % (hours, mins, secs)


class LogTotalSteps(SessionRunHook):
    """docstring for LogTotalSteps"""
    def __init__(
        self,
        batch_size=None,
        train_size=None,
        epochs=None,
        every_n_iter=None,
        every_n_secs=None):

        super(LogTotalSteps, self).__init__()
 
        if ((every_n_iter is None) and (every_n_secs is None)):
            raise ValueError(
                "exactly one of every_n_iter and every_n_secs "
                "must be provided.")
        if every_n_iter is not None and every_n_iter <= 0:
            raise ValueError("invalid every_n_iter=%s." % every_n_iter)

        self._timer = SecondOrStepTimer(
            every_secs=every_n_secs,
            every_steps=every_n_iter)

        self.batch_size = batch_size
        self.train_size = train_size
        self.epochs = epochs

        self.total_steps = self.train_size // self.batch_size * self.epochs

    def begin(self):
        self._timer.reset()
        self._iter_count = 0

    def before_run(self, run_context):  # pylint: disable=unused-argument
        self._should_trigger = self._timer.should_trigger_for_step(
            self._iter_count)

    def _log_total_steps(self):
        original = np.get_printoptions()
        np.set_printoptions(suppress=True)
        elapsed_secs, elapsed_steps = self._timer.update_last_triggered_step(
            self._iter_count)

        steps_to_go = self.total_steps - self._iter_count
        if elapsed_steps is not None and elapsed_secs is not None:
            steps_per_sec = elapsed_steps / elapsed_secs
            ETA = steps_to_go / steps_per_sec
            logging.info("Steps to go: {}, ETA: {}".format(
                steps_to_go, humanize_time(ETA)))
        else:
            logging.info("Steps to go: {}".format(steps_to_go))

        np.set_printoptions(**original)

    def after_run(self, run_context, run_values):
        _ = run_context
        _ = run_values
        if self._should_trigger:
            self._log_total_steps()
        self._iter_count += 1