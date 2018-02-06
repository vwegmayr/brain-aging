import tensorflow as tf

class PrintAndLogTensorHook(tf.train.LoggingTensorHook):
    def __init__(self, estimator, *args, **kwargs):
        super(PrintAndLogTensorHook, self).__init__(
            *args,
            **kwargs
        )
        self.estimator = estimator

    def _log_tensors(self, tensor_values):
        """
        tensor_values is a dict {string => tensor_value }
        """
        super(PrintAndLogTensorHook, self)._log_tensors(tensor_values)
        self.estimator.training_log_values(tensor_values)
