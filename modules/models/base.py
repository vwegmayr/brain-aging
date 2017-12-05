import tensorflow as tf
import multiprocessing
import os
import nibabel as nib
import numpy as np

from sklearn.utils.validation import check_is_fitted
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
from modules.models.utils import print
from tensorflow.python.estimator.export.export import (
    build_raw_serving_input_receiver_fn as input_receiver_fn)


class BaseTF(ABC, BaseEstimator, TransformerMixin):
    """docstring for BaseTF"""
    lock = multiprocessing.Lock()
    num_instances = 0

    def __init__(self, input_fn_config, config, params):
        super(BaseTF, self).__init__()
        self.input_fn_config = input_fn_config
        self.config = config
        self.params = params
        self.id = id

        self._restore_path = None

        with BaseTF.lock:
            self.instance_id = BaseTF.num_instances
            BaseTF.num_instances += 1

    def fit(self, X, y):
        with BaseTF.lock:
            config = self.config
            if BaseTF.num_instances > 1:
                config["model_dir"] = os.path.join(
                    config["model_dir"],
                    "inst-" + str(self.instance_id))

        self.estimator = tf.estimator.Estimator(
            model_fn=self.model_fn,
            params=self.params,
            config=tf.estimator.RunConfig(**config))

        tf.logging.set_verbosity(tf.logging.INFO)
        try:
            self.estimator.train(input_fn=self.input_fn(X, y))
        except KeyboardInterrupt:
            print("\nEarly stop of training, saving model...")
            self.export_estimator(
                input_shape=list(X.shape[1:]),
                input_dtype=X.dtype.name)
        else:
            self.export_estimator(
                input_shape=list(X.shape[1:]),
                input_dtype=X.dtype.name)

        return self

    def predict(self, X, head="predictions"):
        predictor = tf.contrib.predictor.from_saved_model(self._restore_path)
        return predictor({"X": X})[head]

    def predict_proba(self, X):
        return self.predict(X, head="probabs")

    def input_fn(self, X, y):
        return tf.estimator.inputs.numpy_input_fn(
            x={"X": X},
            y=y,
            **self.input_fn_config)

    def set_save_path(self, save_path):
        self.save_path = save_path
        if self._restore_path is None:
            self.config["model_dir"] = save_path

    def export_estimator(self, input_shape, input_dtype):
        feature_spec = {"X": tf.placeholder(
            shape=[None] + input_shape,
            dtype=input_dtype)}
        receiver_fn = input_receiver_fn(feature_spec)
        self._restore_path = self.estimator.export_savedmodel(
            self.save_path,
            receiver_fn)
        print("Model saved to {}".format(self._restore_path))

    @abstractmethod
    def score(self, X, y):
        pass

    @abstractmethod
    def model_fn(self, features, labels, mode, params, config):
        pass

    def __getstate__(self):
        state = self.__dict__.copy()

        for key, val in list(state.items()):
            if "tensorflow" in getattr(val, "__module__", "None"):
                del state[key]

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class BaseTracker(BaseTF):
    """Exension to BaseTF to enable fiber tracking."""

    def __init__(self, input_fn_config, config, params, track_config=None):
        super(BaseTracker, self).__init__(input_fn_config, config, params)
        self.track_config = track_config
        self.wm_mask = nib.load(track_config['wm_mask']).get_data()
        self.nii = nib.load(track_config['nii_file']).get_data()

    def track(self):
        """Generate the tracktography with the current model on the given brain."""
        # Check model
        check_is_fitted(self, ['estimator'])

        # Get brain information
        brain_file = nib.load(self.track_config['nii_file'])
        brain_data = brain_file.get_data()
        brain_header = brain_file.header.structarr
        brain_size = brain_data.shape
        voxel_size = brain_header["pixdim"][1:4]

        if self.track_config['seeds']:
            seeds = self._seeds_from_wm_mask()

        self.tractography = []         # The final result will be here
        self.ongoing_fibers = seeds    # Fibers that are still under construction. At first seeds.

    def _build_next_X(self, last_incoming):
        """Builds the next X-batch to be fed to the model.

        The X-batch created continues the streamline based on the outgoing directions obtained at
        the previous step.

        Returns:
            next_X: The next batch of point values (blocks, incoming, centers).
        """
        pass

    def _seeds_from_wm_mask(self):
        """Compute the seeds for the streamlining from the white matter mask.

        This is invoked only if no seeds are specified.
        The seeds are selected on the interface between white and gray matter, i.e. they are the
        white matter voxels that have at least one gray matter neighboring voxel.
        These points are furthermore perturbed with some gaussian noise to have a wider range of
        starting points.

        Returns:
            seeds: The list of voxel that are seeds.
        """
        # Take te border voxels as seeds
        seeds = self._find_borders()
        print("Number of seeds on the white matter mask:", len(seeds))
        print("Number of requested seeds:", self.track_config['n_fibers'])
        new_idxs = np.random.choice(len(seeds), self.track_config['n_fibers'], replace=True)
        new_seeds = [[seeds[i] + np.clip(np.random.normal(0, 0.25, 3), -0.5, 0.5)]
                     for i in new_idxs]
        return new_seeds

    def _find_borders(self, order=1):
        """Find the wm-gm interface points.

        Args:
            order: How far from the center voxel to look for differen voxels. Default 1.
        Return:
            seeds: The seeds generated from the white matter mask
        """
        dim = self.mask.shape
        borders = []
        for x in range(order, dim[0] - order):
            for y in range(order, dim[1] - order):
                for z in range(order, dim[2] - order):
                    if self.mask[x, y, z] == 1:
                        window = self.mask[x - order:x + 1 + order,
                                           y - order:y + 1 + order,
                                           z - order:z + 1 + order]
                        if not np.all(window):
                            borders.append(np.array([x, y, z]))
        return borders

    def _is_border(self, coord):
        """Check if the voxel is on the white matter border.

        Args:
            coord: Numpy ndarray containing the [x, y, z] coordinates of the point.

        Returns:
            True if the [x, y, z] point is on the border.
        """
        pass
