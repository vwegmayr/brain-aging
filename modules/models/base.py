import tensorflow as tf
import multiprocessing
import os
import nibabel as nib
import numpy as np

from sklearn.utils.validation import check_is_fitted
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin

from modules.models.utils import print, save_fibers, np_placeholder
from modules.models.example_loader import PointExamples, aff_to_rot

from tensorflow.python.estimator.export.export import (
    build_raw_serving_input_receiver_fn as input_receiver_fn)


def train_size(X):
    if isinstance(X, np.ndarray):
        return X.shape[0]
    elif isinstance(X, dict):
        for key, val in X.items():
            if isinstance(val, np.ndarray):
                return val.shape[0]


def input_fn(X, y=None, input_fn_config={}):
    if isinstance(X, np.ndarray):
        X_ = {"X": X}
    elif isinstance(X, dict):
        for key, val in X.items():
            if not isinstance(val, np.ndarray):
                raise ValueError(("Expected values of dict X "
                    "as type np.ndarray, got {} for key {}")
                    .format(type(val), key))
        X_ = X
    else:
        raise ValueError("Expected input X instance of type "
            "ndarray or dict, got {}".format(type(X)))

    return tf.estimator.inputs.numpy_input_fn(x=X_, y=y,
        **input_fn_config)


def feature_spec_from(X):
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
    return feature_spec


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
        if "LogTotalSteps" in self.params["hooks"]:
            self.params["hooks"]["LogTotalSteps"]["batch_size"] = self.input_fn_config["batch_size"]
            self.params["hooks"]["LogTotalSteps"]["epochs"] = self.input_fn_config["num_epochs"]
            self.params["hooks"]["LogTotalSteps"]["train_size"] = train_size(X)

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

        self.feature_spec = feature_spec_from(X)

        tf.logging.set_verbosity(tf.logging.INFO)
        try:
            self.estimator.train(input_fn=input_fn(X, y, self.input_fn_config))
        except KeyboardInterrupt:
            print("\nEarly stop of training, saving model...")
            self.export_estimator()
        else:
            self.export_estimator()

        return self

    def predict(self, X, head="predictions"):
        check_is_fitted(self, ["_restore_path"])

        predictor = tf.contrib.predictor.from_saved_model(self._restore_path)

        if isinstance(X, np.ndarray):
            return predictor({"X": X})[head]
        elif isinstance(X, dict):
            return predictor(X)[head]

    def predictor(self, feature_spec):
        if self._restore_path is not None:
            return tf.contrib.predictor.from_saved_model(self._restore_path)

        elif self.estimator.latest_checkpoint() is not None:
            return tf.contrib.predictor.from_estimator(
                self.estimator,
                input_receiver_fn(self.feature_spec)
            )

        else:
            return None

    def predict_proba(self, X):
        return self.predict(X, head="probabs")

    def set_save_path(self, save_path):
        self.save_path = save_path
        if self._restore_path is None:
            self.config["model_dir"] = save_path

    def export_estimator(self):
        receiver_fn = input_receiver_fn(self.feature_spec)
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


        def remove_tensorflow(state):
            for key, val in list(state.items()):
                if "tensorflow" in getattr(val, "__module__", "None"):
                    del state[key]
                elif isinstance(val, dict):
                    remove_tensorflow(val)

        remove_tensorflow(state)

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class BaseTracker(BaseTF):
    """Exension to BaseTF to enable fiber tracking.

    Extend this class to implement the methods and use for tracking.
    """

    def __init__(self, input_fn_config, config, params):
        super(BaseTracker, self).__init__(input_fn_config, config, params)

    def fit(self, X, y):

        assert isinstance(X, dict)

        self.n_incoming = int(X["incoming"].shape[1] / 3)
        self.block_size = X["blocks"].shape[1]

        super(BaseTracker, self).fit(X, y)

    def predict(self, X, args):
        """Generate the tracktography with the current model on the given brain."""
        # Check model
        check_is_fitted(self, ["n_incoming", "block_size"])
        assert isinstance(X, dict)

        #predictor = tf.contrib.predictor.from_saved_model(self._restore_path)

        predictor = self.predictor(self.feature_spec)

        self.args = args

        try:
            self.wm_mask = X['mask']
            self.nii = X['dwi']
            if 'max_fiber_length' in args:
                self.max_fiber_length = args.max_fiber_length
            else:
                self.max_fiber_length = 400
        except KeyError as err:
            print("KeyError: {}".format(err))

        # Get brain information
        self.brain_data = X['dwi']

        # If no seeds are specified, build them from the wm mask
        if 'seeds' not in self.args:
            seeds = self._seeds_from_wm_mask()

        # The final result will be here
        self.tractography = []
        # Fibers that are still under construction. At first seeds.
        self.ongoing_fibers = seeds

        if predictor is not None:
            # Start tractography generation
            if 'reseed_endpoints' in self.args:
                self._generate_masked_tractography(
                    self.args.reseed_endpoints,
                    affine=X["header"]["vox_to_ras"],
                    predictor=predictor)
            else:
                self._generate_masked_tractography(
                    affine=X["header"]["vox_to_ras"],
                    predictor=predictor)

            # Save the Fibers
            fiber_path = os.path.join(self.save_path, "fibers.trk")
            save_fibers(self.tractography, X["header"], fiber_path)

    def _generate_masked_tractography(
            self,
            reseed_endpoints=False,
            affine=None,
            predictor=None):
        """Generate the tractography using the white matter mask.

        Args:
            reseed_endpoints: Boolean. If True, use the end points of the fibers
                produced to generate another tractography. This is to symmetrize
                the process.
        """

        i = 0
        while self.ongoing_fibers:
            i += 1
            predictions = predictor(self._build_next_X(affine))["predictions"]
            directions = self.get_directions_from_predictions(predictions, affine)

            # Update the positions of the fibers and check if they are still ongoing
            cur_ongoing = []
            for j, fiber in enumerate(self.ongoing_fibers):
                new_position = fiber[-1] + directions[j] * self.args.step_size

                if i == 1 and self._is_border(fiber[-1] + directions[j]):
                    # First step is ambiguous and leads into boarder -> flip it.
                    new_position = fiber[-1] - directions[j] * self.args.step_size

                # Only continue fibers inside the boundaries and short enough
                if self._is_border(new_position) or \
                        i * self.args.step_size > self.max_fiber_length:
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
            self._generate_masked_tractography(reseed_endpoints=False, affine=affine)

    def _build_next_X(self, affine):
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
            incoming_point = np.zeros((self.n_incoming, 3))
            outgoing = np.zeros(3)
            for i in range(min(self.n_incoming, len(fiber)-1)):
                incoming_point[i] = fiber[-i - 2]
            sample = PointExamples.build_datablock(self.brain_data,
                                                   self.block_size,
                                                   center_point,
                                                   incoming_point,
                                                   outgoing,
                                                   label_type,
                                                   affine)
            X_sample = {
                'centers': sample['center'],
                'incoming': sample['incoming'],
                'blocks': sample['data_block']
            }
            # Add example to examples by appending individual lists
            for key, cur_list in X.items():
                cur_list.append(X_sample[key])

        for key, _ in X.items():
            X[key] = np.array(X[key])

        return X

    @abstractmethod
    def get_directions_from_predictions(self, predictions, affine):
        """Computes fiber directions form the predictions of the network.

        Method to be extended in subclasses. By extending this the outputs of
        different types of networks can be used in the same way.

        Args:
            predictions: The output of the neural network model.
            affine: The affine transformation for the voxel space.
        Returns:
            directions: The fiber directions corresponding to the predictions.
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
        print("Number of requested seeds:", self.args.n_fibers)
        new_idxs = np.random.choice(len(seeds), self.args.n_fibers, replace=True)
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


class DeterministicTracker(BaseTracker):
    """This is the base model for deterministic tracking.

    A model does deterministic tracking when its output is the direction of the
    fiber (given the possible different inputs), not a probablity distribution
    (see ProbabilisticTracker).
    """

    def get_directions_from_predictions(self, predictions, affine):
        """Compute the direction of the fibers from the deterministic predict.
        """
        predictions = aff_to_rot(affine).dot(predictions.T).T
        return predictions
