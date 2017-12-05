"""This module contains functionality to load tractography training data.

Credit goes to Viktor Wegmayr

Todo:
    Update doc
"""

import os

import numpy as np
import nibabel as nib
from .utils import extract_direction
import functools


print = functools.partial(print, flush=True)


class Examples(object):
    """Base Class for loading tractography training samples.

      This class provides functionality to create blocks of diffusion-data and
      the associated fiber information. The diffusion data block represents the
      input to any learning algorithm, whereas the fiber information serves as
      label.

      Classes derived from this base class handle different forms of input and
      labels. For instance, the input can be raw diffusion measurements or
      derived representations such as diffusion tensor or spherical harmonics.
      Labels describe the local fiber flow which is the subject of prediction.

      Subclasses:
        PointExamples

      Attributes:
        fibers: List of streamlines. Each streamline is a list
          with shape (fiber_length,3) which contains the x,y,z coordinates of
          each point in the fiber.
        fiber_header: Struct array with info about the loaded track file. See
          http://trackvis.org/docs/?subsect=fileformat for more information.
        brain_file: Proxy to the diffusion data file, which is assumed to be of
          nifti format.
        brain_data: MemMap to the diffusion data stored in the nifti file.
        brain_header: Struct array with information about the loaded diffusion
          data file. See https://brainder.org/2012/09/23/the-nifti-file-format/
          for more information.
        voxel_size: List which contains the voxel spacing in x, y, z directions.
          Units are Millimeter.
        block_size: Integer which indicates the entire length of the diffusion
          data block in one dimension. E.g. if 7x7x7 blocks are considered,
          then the block_size is 7. Should be odd.
        train_labels: List which contains all training fiber labels which are
          parsed from the track file. Each label is a dictionary which keys depend
          on the subclass.
        eval_labels: List which contains all evaluation fiber labels which are
          parsed from the track file. Each label is a dictionary which keys depend
          on the subclass.
        block_length: Integer which indicates half the block_size minus one.
          E.g. if 7x7x7 blocks are considered, the block_length is 3, i.e. the
          distance from the center in each direction in voxels.
        voxel_dimension: List as x,y,z dimensions of brain data.
      """

    def __init__(self, nii_file, trk_file, block_size, num_eval_examples):
        """Load the input files and initialize fields.

        Args:
          nii_file: Path to the nifti file which is used as diffusion data
            input.
          trk_file: Path to the trackvis file which is used for the labels,
            should be derived from the data represented in the niiFile.
          block_size: Integer (odd) which indicates the desired data block
            size.
          num_eval_examples: Integer which indicates approximate number of
            evaluation examples (and therefore labels) loaded from the track
            file. Actual amount of evaluation examples can vary slightly
            because of adding whole fibers at a time.
        """
        self.brain_file = nib.load(nii_file)
        self.brain_data = self.brain_file.get_data()
        self.brain_header = self.brain_file.header.structarr
        self.voxel_size = self.brain_header["pixdim"][1:4]
        self.block_size = block_size
        self.voxel_dimension = np.shape(self.brain_data)[3]
        if trk_file:
            self.fibers, self.fiber_header = nib.trackvis.read(trk_file, points_space="voxel")
            self.fibers = [fiber[0] for fiber in self.fibers]
            self.train_labels, self.eval_labels = self.initialize_labels(num_eval_examples)
        else:
            self.fibers, self.fiber_header = None, None
            self.train_labels, self.eval_labels = [], []
        self.eval_set = None

    def get_train_batch(self, requested_num_examples):
        """Return a dictionary of examples.

        Main method for external applications.

        Args:
          requested_num_examples: Integer which indicates desired number of
            examples. Should be smaller or equal to num_train_examples else
            warning is raised and num_train_examples are returned.
        Returns:
          A dictionary with keys "center", "incoming", "outgoing" and
          "data_block". Each value is a list of length requested_num_examples.
          The i-th element of e.g. list "dataBlock" contains the data_block for
          the i-th example:
          examples["center"][i] = [x,y,z] or one_hot code
          examples["incoming"][i] = [x,y,z] or one_hot code
          examples["outgoing"][i] = [x,y,z] or one_hot code
          examples["data_block"][i] = np.array
        """
        pass

    def get_eval_set(self):
        """Return the evaluation set.

        Returns:
          A dictionary of evaluation examples. The structure is the same as
          for a training batch. The total number of evaluation samples is
          given by num_eval_examples.
        """

    def initialize_labels(self, num_eval_examples):
        """Parse labels from track file.

        For internal use.

        Returns:
          Tuple of two lists of training and evaluation labels. Each label is
          a dictionary which contains information about fiber flow. The keys of
          a label depend on the subclass.
        """
        pass

    @staticmethod
    def points_to_one_hot(center, point):
        """Calculate one-hot code for neighbor voxels.

        For internal use.

        Args:
          center: List [x,y,z] which contains the coordinates of the voxel
            approached or left by a fiber.
          point: List [x,y,z] which contains the coordinates of the neighbor
            voxel from where the center voxel is approached or left.

        Returns:
          Numpy array of shape (27). It encodes either from which neighbor
          voxel the a fiber entered the center voxel or to which neighbor
          voxel the fiber left the center voxel.
        """
        center_voxel = np.round(center).astype(int)

        if not np.array_equal(point, np.zeros(3)):
            point_voxel = np.round(point).astype(int)
            relative = point_voxel - center_voxel
        else:
            relative = np.zeros(3, dtype=np.int64)

        num = 13 + np.dot([1, -3, -9], relative)

        one_hot = np.zeros(27)
        one_hot[num] = 1

        return one_hot

    @staticmethod
    def points_to_relative(_from, to):
        """Calculate relative direction from global coordinates.

        For internal use.

        Args:
          _from: List [x,y,z] which contains the coordinates of the voxel
            starting point of a fiber segment.
          to: List [x,y,z] which contains the coordinates of the voxel
            starting point of a fiber segment

        Returns:
          Numpy array of shape (3) of the relative direction from "_from" to "to".
        """
        if not np.array_equal(_from, np.zeros(3)) and not np.array_equal(to, np.zeros(3)):
            relative = np.asarray(to) - np.asarray(_from)
            return relative / np.linalg.norm(relative)
        else:
            return np.zeros(3)

    @staticmethod
    def build_datablock(data, block_size, center_point, incoming_point, outgoing_point, label_type):
        """Creates an example with all the label information and data added.

        Args:
            data: MemMap to the diffusion data stored in the nifti file.
            block_size: Integer which indicates the entire length of the diffusion
              data block in one dimension. E.g. if 7x7x7 blocks are considered,
              then the block_size is 7. Should be odd.
            center_point: List of [x,y,z] of coordinate where fiber goes though.
            incoming_point: List of [x,y,z] of coordinate where fiber comes from.
            outgoing_point: List of [x,y,z] of coordinate where fiber goes to.
            label_type: String which indicates the desired label type which are
              described in the docstring of PointExamples.

        Returns: A dictionary with keys "center", "incoming", "outgoing" and
          "data_block". Each value is a list of length requested_num_examples.
          example["center"] = np.array [x,y,z] or one_hot code
          example["incoming"] = np.array [x,y,z] or one_hot code
          example["outgoing"] = np.array [x,y,z] or one_hot code
          example["data_block"] = np.array
        """
        example = {}

        voxel = np.round(center_point).astype(int)

        if label_type == "one_hot":
            example["center"] = np.round(center_point).astype(int)
            example["incoming"] = Examples.points_to_one_hot(center_point, incoming_point)
            example["outgoing"] = Examples.points_to_one_hot(center_point, outgoing_point)
        elif label_type == "point":
            example["center"] = np.array(center_point)
            # example["incoming"] = Examples.points_to_relative(incoming_point[0], center_point)  # HACK TRAINING
            example["incoming"] = Examples.points_to_relative(incoming_point[0], center_point) * np.array([1, -1, 1])  # HACK PREDICTION
            for i in range(len(incoming_point) - 1):
                # example["incoming"] = np.append(example["incoming"],  # HACK TRAINING
                #                                 Examples.points_to_relative(incoming_point[i + 1],    #HACK TRAINING
                #                                 incoming_point[i]))  # HACK TRAINING
                example["incoming"] = np.append(example["incoming"],    #HACK PREDICTION
                                                Examples.points_to_relative(incoming_point[i + 1],  #HACK PREDICTION
                                                incoming_point[i]) * np.array([1, -1, 1]))  # HACK PREDICTION
            example["outgoing"] = Examples.points_to_relative(center_point, outgoing_point)

        data_shape = np.shape(data)
        example["data_block"] = np.zeros((block_size,
                                          block_size,
                                          block_size,
                                          data_shape[3]))

        if (voxel[0] < 0 or voxel[0] >= data_shape[0] or
            voxel[1] < 0 or voxel[1] >= data_shape[1] or
            voxel[2] < 0 or voxel[2] >= data_shape[2]):
                print("Warning: voxel out of bounds: ({}, {}, {}), data: (0:{}, 0:{}, 0:{})".format(
                        voxel[0], voxel[1], voxel[2], data_shape[0], data_shape[1], data_shape[2]))
                return example

        block_length = int(np.floor(block_size / 2))

        # Pad data if block is out of bounds
        start = [voxel[0] - block_length,
                 voxel[1] - block_length,
                 voxel[2] - block_length]
        end = [voxel[0] + block_length + 1,
               voxel[1] + block_length + 1,
               voxel[2] + block_length + 1]

        example["data_block"][
            max(-(start[0]), 0):(block_size - max(end[0] - data_shape[0], 0)),
            max(-(start[1]), 0):(block_size - max(end[1] - data_shape[1], 0)),
            max(-(start[2]), 0):(block_size - max(end[2] - data_shape[2], 0)),
            :] = np.array(data[
                max(start[0], 0): min(end[0], data_shape[0]),
                max(start[1], 0): min(end[1], data_shape[1]),
                max(start[2], 0): min(end[2], data_shape[2]),
                :])

        return example


class PointExamples(Examples):
    """Class which represents fiber point examples.

    Todo:
      Update doc
    """

    def __init__(self, nii_file, trk_file, block_size, num_eval_examples, data_corrupt_percent=0.0,
                 min_fiber_length=0, ignore_start_point=False, ignore_stop_point=True,
                 cache_examples=False, last_incoming=1, V1=None):
        """Load the input files and initialize fields."""

        self.min_length = min_fiber_length
        self.ignore_start_point = ignore_start_point
        self.ignore_stop_point = ignore_stop_point
        self.last_incoming = last_incoming
        self.eval_fibers = []
        self.train_generator = None
        self.eval_generator = None
        self.cache_examples = cache_examples
        self.data_corrupt_percent = data_corrupt_percent
        self.V1 = V1

        Examples.__init__(self, nii_file, trk_file, block_size,
                          num_eval_examples)

        self.check_empty_data(warning_only=True)

    def initialize_labels(self, num_eval_examples, augment_reverse_fibers=True):
        print("Loading Fibers...")
        fibers_filtered = []
        for fiber in self.fibers:
            fiber_length_mm = 0
            for j in range(1, len(fiber)):
                fiber_length_mm += np.linalg.norm((fiber[j] - fiber[j - 1]) * self.voxel_size)
                if fiber_length_mm > self.min_length:
                    fibers_filtered.append(fiber)
                    break

        np.random.shuffle(fibers_filtered)
        print("Found {}/{} fibers longer than {}mm".format(len(fibers_filtered), len(self.fibers),
                                                           self.min_length))

        # Subsampling to reduce data
        fibers_filtered = fibers_filtered[::4]

        label_list = []
        eval_labels = []
        for fiber in fibers_filtered:
            for j in range(self.ignore_start_point, len(fiber) - self.ignore_stop_point):
                label = {"center": fiber[j]}

                start = max(j - self.last_incoming, 0)
                end = max(j, 0)
                label["incoming"] = fiber[start:end][::-1]
                label["incoming"] = np.append(label["incoming"],
                    np.zeros((self.last_incoming - len(label["incoming"]), 3)), 0)
                if j == len(fiber) - 1:
                    label["outgoing"] = np.zeros(3)
                else:
                    label["outgoing"] = fiber[j + 1]
                label_list.append(label)
                if augment_reverse_fibers:
                    # TODO: consider ignoring start and end
                    start = min(j + 1, len(fiber))
                    end = min(j + 1 + self.last_incoming, len(fiber))
                    incoming = fiber[start:end]
                    incoming = np.append(incoming,
                                         np.zeros((self.last_incoming - len(incoming), 3)),
                                         0)
                    reverse_label = {"center": label["center"], "incoming": incoming,
                                     "outgoing": label["incoming"][0]}
                    label_list.append(reverse_label)
            if len(eval_labels) == 0 and num_eval_examples > 0:
                self.eval_fibers.append(fiber)
            if len(label_list) >= num_eval_examples and len(eval_labels) == 0 \
                    and num_eval_examples > 0:
                eval_labels = label_list
                label_list = []

        if len(eval_labels) < num_eval_examples:
            print("ERROR: PointExamples: Requested more evaluation examples than available")
            eval_labels = label_list
            label_list = []
        print("finished loading, now shuffle")
        train_labels = label_list
        np.random.shuffle(eval_labels)
        np.random.shuffle(train_labels)
        print("Used {} fibers for training and {} for evaluation".format(
            len(fibers_filtered) - len(self.eval_fibers), len(self.eval_fibers)))
        print("Generated {} train and {} eval fiber labels\n".format(len(train_labels),
                                                                     len(eval_labels)))
        # NOTE: Here is the corruption of the training labels.
        # First, we calculate how many labels have to be corrupted. Then, this number of labels is
        # corrupted by removing the outgoing label and in its place putting a new random one that
        # has been obtained by adding to the 'center' a random unit vector in R3.
        # NOTE: Labels have already been shuffled, so this can be carried on in sequential order.
        if self.data_corrupt_percent > 0.0:
            n_to_corrupt = int(np.floor(len(train_labels) * self.data_corrupt_percent))
            print("DEBUG: Corrupting data. Corruption number is ", n_to_corrupt, "on a total of", len(train_labels))
            for idx in range(n_to_corrupt):
                cur_label = train_labels[idx]
                cur_center = cur_label['center']
                random_v = np.random.normal(size=3)
                random_v = np.divide(random_v, np.linalg.norm(random_v))
                new_outgoing = cur_center + random_v
                cur_label['outgoing'] = new_outgoing
                train_labels[idx] = cur_label  # QUESTION: is this really necessary?
        # Done with the corruption
        return (train_labels, eval_labels)

    def example_generator(self, labels, label_type):
        if label_type not in ["one_hot", "point"]:
            print("ERROR: PointExamples: build_batch: Unknown label_type")

        examples = []
        while True:
            if examples:
                for example in examples:
                    yield example
            else:
                for label in labels:
                    example = Examples.build_datablock(self.brain_data, self.block_size,
                                                       label["center"], label["incoming"],
                                                       label["outgoing"], label_type)
                    yield example
                    if self.cache_examples:
                        examples.append(example)
            if self.cache_examples:
                np.random.shuffle(examples)
            else:
                np.random.shuffle(labels)

    def get_batch(self, generator, requested_num_examples=0):
        """ Return a dictionary of examples.

        Args:
          requested_num_examples: Integer which indicates desired number of
            examples. Should be smaller or equal to num_train_examples else
            warning is raised and num_train_examples are returned.
          generator: Generator from which to pull examples from.

        Returns:
          A dictionary with keys "center", "incoming", "outgoing" and
          "data_block". Each value is a list of length requested_num_examples.
          The i-th element of e.g. list "dataBlock" contains the data_block
          array for the i-th example:
          examples["center"][i] = [x,y,z] or one_hot code
          examples["incoming"][i] = [x,y,z] or one_hot code
          examples["outgoing"][i] = [x,y,z] or one_hot code
          examples["data_block"][i] = np.array
        """
        batch = {
            "center": [],
            "incoming": [],
            "outgoing": [],
            "data_block": []
        }
        for i in range(requested_num_examples):
            example = next(generator)

            # Add example to examples by appending individual lists
            for key, list in batch.items():
                list.append(example[key])
        return batch

    def get_train_batch(self, requested_num_examples, label_type="point"):
        if self.train_generator is None:
            self.train_generator = self.example_generator(self.train_labels, label_type)
        return self.get_batch(self.train_generator, requested_num_examples)

    def get_eval_batch(self, requested_num_examples, label_type="point"):
        if self.eval_generator is None:
            self.eval_generator = self.example_generator(self.eval_labels, label_type)
        return self.get_batch(self.eval_generator, requested_num_examples)

    def get_eval_set(self, label_type="point"):
        # only calculate once
        if self.eval_set is None:
            eval_generator = self.example_generator(self.eval_labels, label_type)
            self.eval_set = self.get_batch(eval_generator, len(self.eval_labels))
        return self.eval_set

    def print_statistics(self):
        print("Statistics for evalution set:")
        eval_set = self.get_eval_set()
        incoming = np.array(eval_set["incoming"])[:, 0:3]
        outgoing = np.array(eval_set["outgoing"])
        dot_prod = np.sum(incoming * outgoing, axis=1)
        dot_loss = 1 - np.average(dot_prod)
        print("Average Dot Loss (1-<incoming, outgoing>): %f" % dot_loss)
        avg_angle = np.average(np.arccos(np.clip(dot_prod, -1, 1))) * 180 / np.pi
        print("Average Angle: %f" % avg_angle)
        if not self.ignore_start_point:
            filter = [not np.array_equal(vec, [0, 0, 0]) for vec in incoming]
            dot_loss_filtered = 1 - np.average(dot_prod[filter])
            print("Loss without starting fibers: %f" % dot_loss_filtered)
            avg_angle = np.average(np.arccos(np.clip(dot_prod[filter], -1, 1))) * 180 / np.pi
            print("Angle without starting fibers: %f" % avg_angle)
        print("-----------------------------")

    def check_alignment(self):
        print("Statistics for eigenvectors of tensors:")

        eval_set = self.get_eval_set()
        outgoing = np.array(eval_set["outgoing"])
        center = np.array(eval_set["center"])
        voxels = np.round(center).astype(int)
        if self.V1 is None:
            if self.voxel_dimension != 6:
                print("Data has wrong dimension to be tensor, skip check")
                return
            tensor = np.array([self.brain_data[voxel[0]][voxel[1]][voxel[2]] for voxel in voxels])
            eigenvec = extract_direction(tensor)
        else:
            eigenvec_data = nib.load(self.V1).get_data()
            eigenvec = [eigenvec_data[voxel[0]][voxel[1]][voxel[2]] for voxel in voxels]

        # take absolute of dot product to ignore ambiguous direction
        dot_prod = np.abs(np.sum(eigenvec * outgoing, axis=1))
        dot_loss = 1 - np.average(dot_prod)
        avg_angle = np.average(np.arccos(np.clip(dot_prod, -1, 1))) * 180 / np.pi
        print("Average Dot Loss (1-<eigenvector, outgoing>): %f" % dot_loss)
        print("Average Angle: %f" % avg_angle)
        print("-----------------------------")

    def check_empty_data(self, warning_only=False, threshold=0.05):
        empty = 0
        data_blocks = self.get_eval_set()["data_block"]
        if len(data_blocks) == 0:
            return
        for data_block in data_blocks:
            if np.isclose(data_block, 0.0).all():
                empty += 1
        percentage = empty / len(data_blocks)
        if warning_only:
            if percentage > threshold:
                print("WARNING: Blocks with empty data: %f" % percentage)
        else:
            print("Blocks with empty data: %f" % percentage)


class UnsupervisedExamples(PointExamples):
    """PointExamples for unsupervised training."""

    def __init__(self, nii_file, trk_file, block_size, num_eval_examples):
        PointExamples.__init__(self, nii_file, trk_file, block_size,
                               num_eval_examples)

    def get_batch(self, generator, requested_num_examples=0):
        """ Return a dictionary of examples.

        Args:
          requested_num_examples: Integer which indicates desired number of
            examples. Should be smaller or equal to num_train_examples else
            warning is raised and num_train_examples are returned.
          label_type: String which indicates the desired label type which are
            described in the docstring of PointExamples.

        Returns:
          A dictionary with keys "center", "incoming", "outgoing" and
          "data_block". Each value is a list of length requested_num_examples.
          The i-th element of e.g. list "dataBlock" contains the data_block
          array for the i-th example:
          examples["center"][i] = [x,y,z] or one_hot code
          examples["incoming"][i] = [x,y,z] or one_hot code
          examples["outgoing"][i] = [x,y,z] or one_hot code
          examples["data_block"][i] = np.array
        """
        batch = {
            "center": [],
            "incoming": [],
            "outgoing": [],
            "data_block": []
        }
        for i in range(requested_num_examples):
            example = next(generator)

            # Add example to examples by appending individual lists
            for key, list in batch.items():
                if key == 'data_block':
                    # still flatten the data blocks
                    list.append(example[key].flatten())
                else:
                    list.append(example[key])
        return batch

    def get_unlabeled_batch(self, generator, requested_num_examples=0):
        examples = []
        for i in range(requested_num_examples):
            example = next(generator)
            examples.append(example["data_block"].flatten())
        return np.array(examples)

    def get_train_batches(self, requested_num_examples):
        """ Return an array of examples.

        Args:
          requested_num_examples: Integer which indicates desired number of
            examples. Should be smaller or equal to num_train_examples else
            warning is raised and num_train_examples are returned.

        Returns:
          An array with the requested number of examples. Each example is a
          flattened array as a list of tensors for the whole cube size, where
          each tensor is represented by the 6 values in it's upper diagonal.
        """
        if self.train_generator is None:
            self.train_generator = self.example_generator(self.train_labels,
                                                          "point")
        return self.get_unlabeled_batch(self.train_generator,
                                        requested_num_examples)

    def get_eval_set(self, label_type="point", unlabeled=False):
        """ Return evaluation examples including labels for ground truth.

        Args:
          num: number of examples. If left to None, all evaluation examples are
               returned
          label_type: String which indicates the desired label type which are
            described in the docstring of PointExamples.

        Returns:
          A dictionary with keys "center", "incoming", "outgoing" and
          "data_block". Each value is a list of length requested_num_examples.
          The i-th element of e.g. list "dataBlock" contains the data_block
          array for the i-th example:
          examples["center"][i] = [x,y,z] or one_hot code
          examples["incoming"][i] = [x,y,z] or one_hot code
          examples["outgoing"][i] = [x,y,z] or one_hot code
          examples["data_block"][i] = np.array
        """
        # only calculate once
        if unlabeled:
            if (not hasattr(self, 'unlabeled_eval_set')) or \
                    self.unlabeled_eval_set is None:
                eval_generator = self.example_generator(self.eval_labels,
                                                        "point")
                self.unlabeled_eval_set = self.get_unlabeled_batch(
                    eval_generator, len(self.eval_labels))
            ret = self.unlabeled_eval_set
        else:
            if self.eval_set is None:
                eval_generator = self.example_generator(self.eval_labels,
                                                        label_type)
                self.eval_set = self.get_batch(eval_generator,
                                               len(self.eval_labels))
            ret = self.eval_set
        return ret


class TestExamples(object):
    """Usage Demonstration of the Examples class

      Make sure you put valid "tensor.nii" and "fibers.trk" files in the same
      directory as this module.
      """

    def __init__(self):
        # TODO: rework
        # Create a new PointExamples instance
        path = str(os.path.dirname(
            os.path.abspath(__file__)).split("example_loader")[0]) + "data/"
        pt_ex = PointExamples(
            path + "tensor.nii",
            path + "fibers.trk",
            block_size=3,
            num_eval_examples=1)
        print("Created PointExamples instance with blocksize 3!")

        # Access interesting attributes
        print("num_train_examples: {}".format(pt_ex.num_train_examples))
        print("num_fibers: {}".format(pt_ex.num_fibers))

        # Check that the initial exampleState is indeed zero
        print("Initial example_state: {}".format(pt_ex.example_state))

        # Get a first one-hot point example
        ex1 = pt_ex.get_train_batch(1, label_type="one_hot")
        print("Got one example!")

        # Now the exampleState is one
        print("Now the exampleState is: {}".format(pt_ex.example_state))

        print("Content of the first example:")
        print("center: {}".format(ex1["center"]))
        print("incoming: {}".format(ex1["incoming"]))
        print("outgoing: {}".format(ex1["outgoing"]))
        print("data_block type: {}".format(type(ex1["data_block"][0])))
        print("data_block shape: {}".format(ex1["data_block"][0].shape))


if __name__ == "__main__":
    pass
