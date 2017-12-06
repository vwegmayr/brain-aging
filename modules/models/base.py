import tensorflow as tf
import multiprocessing
import os
import nibabel as nib
import numpy as np

from sklearn.utils.validation import check_is_fitted
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin

from modules.models.utils import print, save_fibers, np_placeholder
from modules.models.example_loader import PointExamples

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
            self.export_estimator(X)
        else:
            self.export_estimator(X)

        return self

    def predict(self, X, head="predictions"):
        predictor = tf.contrib.predictor.from_saved_model(self._restore_path)
        return predictor({"X": X})[head]

    def predict_proba(self, X):
        return self.predict(X, head="probabs")

    def input_fn(self, X, y):
        if isinstance(X, np.ndarray):
            X_ = {"X": X}
        elif isinstance(X, dict):
            X_ = X
        else:
            raise ValueError("Expected input X instance of type "
                "ndarray or dict, got {}".format(type(X)))

        return tf.estimator.inputs.numpy_input_fn(
            x=X_,
            y=y,
            **self.input_fn_config)

    def set_save_path(self, save_path):
        self.save_path = save_path
        if self._restore_path is None:
            self.config["model_dir"] = save_path

    def export_estimator(self, X):
        feature_spec = {}
        if isinstance(X, np.ndarray):
            feature_spec["X"] = np_placeholder(X)
        elif isinstance(X, dict):
            for key, val in X.items():
                if isinstance(val, np.ndarray):
                    feature_spec[key] = np_placeholder(val)
                else:
                    raise ValueError("Expected X to be dict to ndarray, "
                        "got key {} which is {}.".format(key, type(val)))

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

    def __init__(self, input_fn_config, config, params):
        super(BaseTracker, self).__init__(input_fn_config, config, params)


    def track(self, track_config):
        """Generate the tracktography with the current model on the given brain."""
        # Check model
        #check_is_fitted(self, ['estimator'])
        self.track_config = track_config

        try:
            self.wm_mask = nib.load(track_config['wm_mask']).get_data()
            self.nii = nib.load(track_config['nii_file']).get_data()
            if 'max_fiber_length' in track_config:
                self.max_fiber_length = track_config['max_fiber_length']
            else:
                self.max_fiber_length = 400
        except KeyError as err:
            print("KeyError: {}".format(err))

        # Get brain information
        brain_file = nib.load(self.track_config['nii_file'])
        self.brain_data = brain_file.get_data()

        # If no seeds are specified, build them from the wm mask
        if 'seeds' not in self.track_config:
            seeds = self._seeds_from_wm_mask()

        self.tractography = []         # The final result will be here
        self.ongoing_fibers = seeds    # Fibers that are still under construction. At first seeds.

        # Start tractography generation
        if 'reseed_endpoints' in self.track_config:
            self._generate_masked_tractography(self.track_config['reseed_endpoints'])
        else:
            self._generate_masked_tractography()

        # Now in self.tractography there are all the finished fibers
        # Build the header for the new fibers
        new_header = nib.trackvis.empty_header()
        affine = brain_file.affine
        nib.trackvis.aff_to_hdr(affine, new_header, True, True)
        new_header["dim"] = brain_file.header.structarr["dim"][1:4]
        # Save the Fibers
        if 'out_name' in self.track_config:
            save_fibers(self.tractography, new_header, self.track_config['out_name'])
        else:
            save_fibers(self.tractography, new_header)


    def _generate_masked_tractography(self, reseed_endpoints=False):
        """Generate the tractography using the white matter mask.

        Args:
            reseed_endpoints: Boolean. If True, use the end points of the fibers produced to
                generate another tractography. This is to symmetrize the process.
        """

        i = 0
        while self.ongoing_fibers:
            i += 1
            # TODO: WARNING: there is a HACK here in the origninal code. Probably the problem with
            # the tracto alignment.
            predictions = self.predict(self._build_next_X())

            # Update the positions of the fibers and check if they are still ongoing
            cur_ongoing = []
            for j, fiber in enumerate(self.ongoing_fibers):
                new_position = fiber[-1] + predictions[j] * self.track_config['step_size']

                if i == 1 and self.is_border(fiber[-1] + predictions[j]):
                    # First step is ambiguous and leads into boarder -> flip it.
                    new_position = fiber[-1] - predictions[j] * self.track_config['step_size']

                # Only continue fibers inside the boundaries and short enough
                if self.is_border(new_position) or \
                        i * self.track_config['step_size'] > self.max_fiber_length:
                    self.tractography.append(fiber)
                else:
                    fiber.append(new_position)
                    cur_ongoing.append(fiber)
            self.ongoing_fibers = cur_ongoing

            end = "\r"
            if i % 25 == 0:
                end = "\n"
            print("Round num:", '%4d' % i, "; ongoing:", '%7d' % len(self.ongoing_fibers),
                  "; completed:", '%7d' % len(self.tractography), end=end)

        if reseed_endpoints:
            ending_seeds = [[fiber[-1]] for fiber in self.tractography]
            self.ongoing_fibers = ending_seeds
            self._generate_masked_tractography(reseed_endpoints=False)

    def _build_next_X(self):
        """Builds the next X-batch to be fed to the model.

        The X-batch created continues the streamline based on the outgoing directions obtained at
        the previous step.

        Returns:
            next_X: The next batch of point values (blocks, incoming, centers).
        """
        label_type = "point"
        X = {
            'centers': [],
            'incoming': [],
            'blocks': []
        }

        for fiber in self.ongoing_fibers:
            center_point = fiber[-1]
            incoming_point = np.zeros((self.track_config['n_last_incoming'], 3))
            outgoing = np.zeros(3)
            for i in range(min(self.track_config['n_last_incoming'], len(fiber)-1)):
                incoming_point[i] = fiber[-i - 2]
            sample = PointExamples.build_datablock(self.brain_data,
                                                   self.track_config['block_size'],
                                                   center_point,
                                                   incoming_point,
                                                   outgoing,
                                                   label_type)
            X_sample = {
                'centers': sample['center'],
                'incoming': sample['incoming'],
                'blocks': sample['data_block']
            }
            # Add example to examples by appending individual lists
            for key, cur_list in X.items():
                cur_list.append(X_sample[key])
        return X

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
        dim = self.wm_mask.shape
        borders = []
        for x in range(order, dim[0] - order):
            for y in range(order, dim[1] - order):
                for z in range(order, dim[2] - order):
                    if self.wm_mask[x, y, z] == 1:
                        window = self.wm_mask[x - order:x + 1 + order,
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
        coord = np.round(coord).astype(int)

        lowerbound_condition = coord[0] < 0 or coord[1] < 0 or coord[2] < 0
        upperbound_condition = coord[0] >= self.wm_mask.shape[0] or \
                               coord[1] >= self.wm_mask.shape[1] or \
                               coord[2] >= self.wm_mask.shape[2]

        # Check if out of image dimensions
        if lowerbound_condition or upperbound_condition:
            return True
        # Check if out of white matter area
        return np.isclose(self.wm_mask[coord[0], coord[1], coord[2]], 0.0)
