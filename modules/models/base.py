import tensorflow as tf
import multiprocessing
import os
import nibabel as nib
import numpy as np
from distutils.dir_util import copy_tree

from sklearn.utils.validation import check_is_fitted
import abc, six
from abc import abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin

from modules.models.utils import custom_print, save_fibers, np_placeholder
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


@six.add_metaclass(abc.ABCMeta)
class BaseTF(BaseEstimator, TransformerMixin):
    """docstring for BaseTF"""
    lock = multiprocessing.Lock()
    num_instances = 0

    def __init__(self, input_fn_config, config, params):
        super(BaseTF, self).__init__()
        self.input_fn_config = input_fn_config
        self.config = config
        self.params = params
        self.feature_spec = None

        self._restore_path = None

        with BaseTF.lock:
            self.instance_id = BaseTF.num_instances
            BaseTF.num_instances += 1

    def fit(self, X, y):
        if "hooks" in self.params and "LogTotalSteps" in self.params["hooks"]:
            self.params["hooks"]["LogTotalSteps"]["batch_size"] = self.input_fn_config["batch_size"]
            self.params["hooks"]["LogTotalSteps"]["epochs"] = self.input_fn_config["num_epochs"]
            self.params["hooks"]["LogTotalSteps"]["train_size"] = train_size(X)

        with BaseTF.lock:
            config = self.config
            if BaseTF.num_instances > 1:
                config["model_dir"] = os.path.join(
                    config["model_dir"],
                    "inst-" + str(self.instance_id))

        self.est_config = config
        self.estimator = tf.estimator.Estimator(
            model_fn=self.model_fn,
            params=self.params,
            config=tf.estimator.RunConfig(**config))

        if self.feature_spec is None:
            self.feature_spec = feature_spec_from(X)

        tf.logging.set_verbosity(tf.logging.ERROR)
        try:
            self.fit_main_training_loop(X, y)
        except KeyboardInterrupt:
            custom_print("\nEarly stop of training, saving model...")
            self.export_estimator()
            return self
        else:
            self.export_estimator()
            return self

    def fit_main_training_loop(self, X, y):
        self.estimator.train(
            input_fn=self.gen_input_fn(X, y, True, self.input_fn_config)
        )
        evaluation_fn = self.gen_input_fn(X, y, False, self.input_fn_config)
        if evaluation_fn is None:
            custom_print("No evaluation data available - skipping evaluation.")
            return
        evaluation = self.estimator.evaluate(input_fn=evaluation_fn)
        custom_print(evaluation)

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
        if "load_model" in self.config:
            copy_tree(
                self.config["load_model"],
                save_path
            )

            del self.config["load_model"]

        if self._restore_path is None:
            self.config["model_dir"] = save_path

    def export_estimator(self):
        receiver_fn = input_receiver_fn(self.feature_spec)
        self._restore_path = self.estimator.export_savedmodel(
            self.save_path,
            receiver_fn)
        custom_print("Model saved to {}".format(self._restore_path))

    @abstractmethod
    def score(self, X, y):
        pass

    @abstractmethod
    def model_fn(self, features, labels, mode, params, config):
        pass

    def gen_input_fn(self, X, y=None, train=True, input_fn_config={}):
        """
        Returns a function that when called returns the data iterator.
        To support various datasets (train/validation/...), you need to
        reimplement it in a child class
        """
        if not train:
            return None
        if isinstance(X, np.ndarray):
            X_ = {"X": X}
        elif isinstance(X, dict):
            for key, val in X.items():
                if not isinstance(val, np.ndarray):
                    raise ValueError((
                        "Expected values of dict X "
                        "as type np.ndarray, got {} for key {}")
                        .format(type(val), key))
            X_ = X
        else:
            raise ValueError(
                "Expected input X instance of type "
                "ndarray or dict, got {}".format(type(X)))

        return tf.estimator.inputs.numpy_input_fn(
            x=X_,
            y=y,
            **input_fn_config
        )

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
            custom_print("KeyError: {}".format(err))

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

                if i == 1 and self._is_border(new_position):
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
            custom_print("Round num:", '%4d' % i, "; ongoing:", '%7d' % len(self.ongoing_fibers),
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
        custom_print("Number of seeds on the white matter mask:", len(seeds))
        custom_print("Number of requested seeds:", self.args.n_fibers)
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
    """Base model for deterministic tracking.

    A model does deterministic tracking when its output is the direction of the
    fiber (given the possible different inputs), not a probablity distribution
    (see ProbabilisticTracker).
    """

    def get_directions_from_predictions(self, predictions, affine):
        """Compute the direction of the fibers from the deterministic predict.
        """
        predictions = aff_to_rot(affine).dot(predictions.T).T
        return predictions


class ProbabilisticTracker(BaseTracker):
    """Base model for probabilistic tracking.

    This model assumes that the network does not output a direction but a
    probability distribution over the possible directions. In this case, the
    distribution is the Fisher-Von Mises distribution, with parameters [mu, k],
    where mu is the 3-dimensional mean-direciton vector and k is the
    concentration parameter.
    """

    def get_directions_from_predictions(self, predictions, affine):
        # NOTE: WE MUST DECIDE HERE HOW TO STORE MEAN AND CONCENTRAION IN THE
        # PREDICTIONS
        mu = predictions['mean']
        k = predictions['concentration']
        directions = ProbabilisticTracker.sample_vMF(mu, k)
        return directions

    @staticmethod
    def sample_vMF(mu, k):
        """Sampe from the von Mises-Fisher distribution.

        See "Numerically stable sampling of the von Mises Fisher distribution
        onS2 (and other tricks)".
        https://www.mendeley.com/viewer/?fileId=1d3bb1ab-8211-60fb-218c-f11e1638
        0bde&documentId=7eb942de-6dd9-3af7-b36c-8a9c37b6b6a6

        Args:
            mu: Mean of the distribution. Shape (N, 3).
            k: Concentration of the distribution. Shape (N, 3).
        Returns:
            samples: Samples from the specified vMF distribution. Ndarray of
                shape (N, 3), where N is the number of different distributions.
                A row of the matrix of index j is a sample from the vMF with
                mean mu[j] and concentration k[j]
        """
        # Make them
        mu = np.asarray(mu)
        k = np.asarray(k)
        # Get the dimensions
        n_samples = mu.shape[0]
        if n_samples != k.shape[0]:
            raise ValueError("The means and the concentations must be in the \
                             same number.")
        # Get the values for V and W
        V = ProbabilisticTracker.sample_unif_unit_circle(n_samples)
        W = ProbabilisticTracker._sample_W_values(n_samples, k)
        # Compute the first part of the sampled vector with mean (0, 0, 1)
        factor = np.matrix(np.sqrt(1 - np.square(W))).T
        omega_1 = np.multiply(factor, V)
        # The second part is W itself
        W = np.matrix(W).T
        omega = np.asarray(np.hstack((omega_1, W)))
        # Now apply the rotation to change the mean
        # i.e. rotate from the direction of the z-axis to the mean direction
        reference = np.asarray([[0, 0, 1]] * omega.shape[0])
        rotation = ProbabilisticTracker._rotation_matrices(reference, mu)
        samples = np.matmul(rotation, omega[:, :, np.newaxis])[:, :, 0]
        return samples

    @staticmethod
    def _rotation_matrices(vectors, references):
        """Compute all the rotation matrices from the vectors to the references.

        Args:
            vectors: Array of vectors that have to be rotated to match the
                references.
            references: Array of reference vectors.
        Returns:
            rotations: Array of matrices. Each matrix is the rotation form the
                vector of corresponding index to its reference.
        """
        # TODO: Fix the inefficiency of the for loop to compute the rotation
        # matrices
        rotations_list = []
        for idx in range(vectors.shape[0]):
            rot_mat = ProbabilisticTracker._to_rotation(vectors[idx, :],
                                                        references[idx, :])
            rotations_list.append(rot_mat)
        rotations = np.asarray(rotations_list)
        return np.asarray(rotations)

    @staticmethod
    def _to_corss_skew_symmetric(vec, ref):
        """Finds the skew-symmetric cross-product matrix."""
        v = np.cross(vec, ref)

        cross_mat = np.zeros(shape=(3, 3))
        cross_mat[[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]] = [-v[2],
                                                             v[1],
                                                             v[2],
                                                             -v[0],
                                                             -v[1],
                                                             v[0]]
        return cross_mat

    @staticmethod
    def _to_rotation(vec, ref):
        """Compute rotation matrix from vec to ref.

        NOTE: There must be a better way to do this.
        """
        cross = ProbabilisticTracker._to_corss_skew_symmetric(vec, ref)
        c = np.reshape(np.asarray(np.dot(vec, ref)), newshape=-1)
        square = np.dot(cross, cross)
        R = np.eye(3) + cross + square * (1 / (1 + c))
        return R

    @staticmethod
    def sample_unif_unit_circle(n_samples):
        """Sample form the uniform distribution on the unit circle.

        Args:
            n_samples: Number of samples required.
        Returns:
            samples: (n_samples,2) ndarray.
        """
        unnormed = np.random.randn(n_samples, 2)
        samples = np.divide(unnormed,
                            np.matrix(np.linalg.norm(unnormed, axis=1)).T)
        samples = np.asarray(samples)
        return samples

    @staticmethod
    def _sample_W_values(n_samples, k):
        """Sample the values of W."""
        unif_points = np.random.uniform(size=n_samples)
        return ProbabilisticTracker._inverse_cumulative_distribution(unif_points, k)

    @staticmethod
    def _inverse_cumulative_distribution(unif, k):
        """Numerically stable version of the inverse cumulative distribution
        function to sample from the vMF distribution."""
        w_values = 1 + (1 / k) \
            * np.log(unif + (1 - unif) * np.exp(-2 * k))
        return w_values
