"""
TODOLIST:
- Load data manually (override X/Y)
-
"""
from modules.models.base import BaseTF as TensorflowBaseEstimator


class Estimator(TensorflowBaseEstimator):
    """docstring for Estimator"""

    def score(self, X, y):
        """
        ???
        """
        import ipdb; ipdb.set_trace()

    def model_fn(self, features, labels, mode, params, config):
        """
        https://www.tensorflow.org/extend/estimators#constructing_the_model_fn
        - features:
        - labels:
        - mode:
        - params:
        - config: ???
        """
        import ipdb; ipdb.set_trace()
