import numpy as np
import tensorflow as tf
import sklearn as skl

from sklearn.utils.validation import check_array, check_is_fitted

from modules.models.utils import parse_hooks
from modules.models.base import BaseTF

from tensorflow.python.saved_model.signature_constants import (
    DEFAULT_SERVING_SIGNATURE_DEF_KEY)
from tensorflow.python.estimator.export.export_output import PredictOutput


class ExampleTF(BaseTF):