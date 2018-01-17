from modules.models.base import BaseTF as TensorflowBaseEstimator
from input import train_input


class Estimator(TensorflowBaseEstimator):
    """docstring for Estimator"""

    def score(self, X, y):
        """
        Only used for prediction apparently. Dont need it now.
        """
        assert(False)

    def model_fn(self, features, labels, mode, params, config):
        """
        https://www.tensorflow.org/extend/estimators#constructing_the_model_fn
        - features:
        - labels:
        - mode: {train, evaluate, inference}
        - params:
        - config: ???
        """
        import ipdb; ipdb.set_trace()

    def gen_input_fn(self, X, y=None, input_fn_config={}):
        # TODO: Return "test_input" for testing
        return train_input
