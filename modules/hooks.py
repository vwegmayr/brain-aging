import numpy as np
import datetime
import run

from tensorflow.python.training.session_run_hook import SessionRunHook
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer
from tensorflow.python.platform import tf_logging as logging
from abc import ABC, abstractmethod
from argparse import Namespace


def humanize_time(secs):
    mins, secs = divmod(secs, 60)
    hours, mins = divmod(mins, 60)
    return '%02d:%02d:%02d' % (hours, mins, secs)


def validate_every_n(steps, secs):
    if ((steps is None) and (secs is None)):
        raise ValueError(
            "exactly one of every_n_steps and every_n_secs "
            "must be provided.")
    if steps is not None and steps <= 0:
        raise ValueError("invalid every_n_steps=%s." % steps)


class BaseHook(SessionRunHook, ABC):
    """docstring for BaseHook"""
    def __init__(
        self,
        every_n_steps=None,
        every_n_secs=None):

        super(BaseHook, self).__init__()

        validate_every_n(every_n_steps, every_n_secs)

        self._timer = SecondOrStepTimer(
            every_secs=every_n_secs,
            every_steps=every_n_steps)

    def begin(self):
        self._timer.reset()
        self._iter_count = 0

    def before_run(self, run_context):  # pylint: disable=unused-argument
        self._should_trigger = self._timer.should_trigger_for_step(
            self._iter_count)

    def after_run(self, run_context, run_values):
        _ = run_context
        _ = run_values
        if self._should_trigger:
            self._triggered_action()
        self._iter_count += 1

    @abstractmethod
    def _triggered_action(self):
        pass


class LogTotalSteps(BaseHook):
    """docstring for LogTotalSteps"""
    def __init__(
        self,
        batch_size=None,
        train_size=None,
        epochs=None,
        every_n_steps=None,
        every_n_secs=None):

        super(LogTotalSteps, self).__init__(
            every_n_steps,
            every_n_secs)
 
        self.batch_size = batch_size
        self.train_size = train_size
        self.epochs = epochs

        self.total_steps = self.train_size // self.batch_size * self.epochs

    def _triggered_action(self):
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


class FiberTrackingHook(BaseHook):
    """docstring for FiberTrackingHook"""
    def __init__(
        self,
        tracker=None,
        every_n_steps=None,
        every_n_secs=None,
        start_at_step=1,
        test_set=None,
        n_fibers=1000,
        step_size=1):

        super(FiberTrackingHook, self).__init__(
            every_n_steps,
            every_n_secs)

        self.tracker = tracker
        self.test_set = test_set

        self.args = Namespace(**{
            "n_fibers": n_fibers,
            "step_size": step_size
            })

        self.start_at_step = start_at_step

    def begin(self):
        super(FiberTrackingHook, self).begin()
        self.test_set = run.load_data(self.test_set)

    def _triggered_action(self):

        self._timer.update_last_triggered_step(
            self._iter_count)

        if self._iter_count >= self.start_at_step:
            original = np.get_printoptions()
            np.set_printoptions(suppress=True)
            logging.info("\nSaving Fibers...")
            self.tracker.predict(self.test_set, self.args)
            np.set_printoptions(**original)