import abc
import csv
import glob
import re
import warnings
import numpy as np
import tensorflow as tf
import copy

from . import features as _features


class FileStream(abc.ABC):
    """
    Base class to stream files from disk.
    """
    def __init__(self, stream_config):
        """
        Arg:
            - stream_config: config for streaming specified
              in the yaml configuration file
        """
        self.config = config = stream_config
        self.meta_csv = config["meta_csv"]
        self.meta_id_column = config["meta_id_column"]
        self.batch_size = config["batch_size"]
        self.data_sources_list = config["data_sources"]
        self.seed = config["seed"]
        self.shuffle = config["shuffle"]

        if "feature_collection" in config:
            self.feature_desc = _features.collections[
                config["feature_collection"]
            ].feature_info

        if self.seed is not None:
            np.random.seed(self.seed)
            self.np_random = np.random

        # Parse meta information
        self.file_id_to_meta = self.parse_meta_csv()
        csv_len = len(self.file_id_to_meta)

        self.name_to_data_source = {}
        # Create datasources
        for ds in self.data_sources_list:
            d = DataSource(
                name=ds["name"],
                glob_pattern=ds["glob_pattern"],
                id_from_filename=ds["id_from_filename"]
            )
            self.name_to_data_source[ds["name"]] = d

        # Match files with meta information, only data specified
        # in csv file is used
        self.all_file_paths = []
        n_files_not_used = 0
        for name in self.name_to_data_source:
            ds = self.name_to_data_source[name]
            new_paths = ds.get_file_paths()
            print("{} has {} files".format(name, len(new_paths)))
            # Add path as meta information
            for p in new_paths:
                image_label = ds.get_file_image_label(p)
                if image_label in self.file_id_to_meta:
                    self.file_id_to_meta[p] = copy.deepcopy(
                        self.file_id_to_meta[image_label]
                    )
                    self.file_id_to_meta[p]["file_path"] = p
                    self.all_file_paths.append(p)
                else:
                    n_files_not_used += 1

        print("{} files found but not specified meta csv"
              .format(n_files_not_used))
        print("Number of files: {}".format(len(self.all_file_paths)))
        n_missing = csv_len - len(self.all_file_paths)
        print("Number of files missing: {}".format(n_missing))

        # Group files into tuples
        self.groups = self.group_data()
        self.sample_shape = self.get_sample_shape()

        # Make train-test split based on grouping
        self.make_train_test_split()

    @abc.abstractmethod
    def get_batches(self):
        """
        Produce batches of file groups.
        Return:
            - batches: list of list of groups
        """
        pass

    @abc.abstractmethod
    def group_data(self):
        """
        Group files together that should be streamed together.
        For example groups of two files (i.e. pairs) or groups
        of three files (i.e. triples) can be formed.
        """
        pass

    @abc.abstractmethod
    def make_train_test_split(self):
        """
        Split the previously formed groups in a training set and
        a test set.
        """
        pass

    @abc.abstractmethod
    def load_sample(self):
        pass

    def parse_meta_csv(self):
        """
        Parse the csv file containing the meta information about the
        data.
        """
        meta_info = {}
        with open(self.meta_csv) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                key = row[self.meta_id_column]

                for k in row:
                    if k in self.feature_desc:
                        ty = self.feature_desc[k]["py_type"]
                        row[k] = (ty)(row[k])  # cast to type

                meta_info[key] = row

        return meta_info

    def get_file_path(self, file_id):
        """
        Arg:
            - file_id: id of file
        Return:
            - file path for the given id
        """
        return self.file_id_to_meta[file_id]["file_path"]

    def get_diagnose(self, file_id):
        record = self.file_id_to_meta[file_id]
        if record["health_ad"] == 1:
            return "health_ad"
        if record["healthy"] == 1:
            return "healthy"
        if record["health_mci"] == 1:
            return "health_mci"

        raise ValueError("diagnosis not found for id {}".format(file_id))

    def get_age(self, file_id):
        record = self.file_id_to_meta[file_id]
        return record["age"]

    def get_patient_label(self, file_id):
        record = self.file_id_to_meta[file_id]
        return record["patient_label"]

    def get_sample_shape(self):
        assert len(self.groups) > 0
        some_id = self.groups[0].file_ids[0]
        path = self.get_file_path(some_id)
        sample = self.load_sample(path)

        self.sample_shape = sample.shape
        return sample.shape

    def get_image_label(self, file_id):
        record = self.file_id_to_meta[file_id]
        return record["image_label"]

    def get_input_fn(self, train):
        print("<>>>>>>>>>> called get input fn")
        # TODO: get batch ordering here
        batches = self.get_batches(train)
        groups = [group for batch in batches for group in batch]
        group_size = len(groups[0].file_ids)
        files = [group.file_ids for group in groups]

        feature_keys = next(iter(self.file_id_to_meta.items()))[1].keys()
        port_features = [
            k
            for k in feature_keys
            if (k != _features.MRI) and (k in self.feature_desc)
        ]

        def _read_files(file_ids, label):
            file_ids = [fid.decode('utf-8') for fid in file_ids]
            ret = []
            for fid in file_ids:
                path = self.get_file_path(fid)

                file_features = self.file_id_to_meta[fid]
                image = self.load_sample(path).astype(np.float16)
                ret += [image]

                ret += [
                    file_features[pf]
                    for pf in port_features
                ]
            # print("_read_files {}".format(ret[0] is None))
            return ret  # return list of features

        def _parser(*to_parse):
            sample_shape = self.sample_shape
            el_n_features = 1 + len(port_features)  # sample + csv features
            all_features = {}
            # parse features for every sample in group
            for i in range(group_size):
                self.feature_desc[_features.MRI]["shape"] = sample_shape
                mri_idx = i * el_n_features
                _mri = to_parse[mri_idx]
                ft = {
                    _features.MRI: tf.reshape(_mri, sample_shape),
                }

                ft.update({
                    port_features[i - 1]: to_parse[i]
                    for i in range(mri_idx + 1, mri_idx + el_n_features)
                })
                ft.update({
                    ft_name: d['default']
                    for ft_name, d in self.feature_desc.items()
                    if ft_name not in ft
                })
                el_features = {
                    ft_name + "_" + str(i): tf.reshape(
                        ft_tensor,
                        self.feature_desc[ft_name]['shape']
                    )
                    for ft_name, ft_tensor in ft.items()
                }  # return dictionary of features, should be tensors
                # rename mri_i to X_i
                el_features["X_" + str(i)] = el_features.pop(_features.MRI + "_" + str(i))
                all_features.update(el_features)

            return {
                "X_0": all_features["X_0"]
            }
            # return all_features

        labels = len(files) * [0]  # currently not used
        dataset = tf.data.Dataset.from_tensor_slices(
            tuple([files, labels])
        )

        read_types = group_size * ([tf.float16] + [
            self.feature_desc[fname]["type"]
            for fname in port_features
        ])

        dataset = dataset.map(
            lambda file_ids, label: tuple(tf.py_func(
                _read_files,
                [file_ids, label],
                read_types,
                stateful=False,
                name="read_files"
            )),
            num_parallel_calls=12
        )

        dataset = dataset.map(_parser)
        # dataset = dataset.prefetch(10 * self.config["batch_size"])
        dataset = dataset.batch(batch_size=self.config["batch_size"])

        def _input_fn():
            return dataset.make_one_shot_iterator().get_next()
        return _input_fn


class Group(object):
    """
    Represents files that should be considered as a group.
    """
    def __init__(self, file_ids, is_train=None):
        self.file_ids = file_ids
        self.is_train = is_train

    def get_file_ids(self):
        return self.file_ids

    def __str__(self):
        return str([str(i) for i in self.file_ids])


class DataSource(object):
    """
    Represents a dataset that is located in one folder.
    """
    def __init__(self, name, glob_pattern, id_from_filename):
        """
        Args:
            - name: name of the dataset
            - glob_pattern: regular expression that identifies
              files that should be considered
            - id_from_filename: dictionary containing a regular
              expression identifying valid file names and the ID
              of the group in the regular expression that contains
              the file ID, e.g. 3991 is the ID in 3991_aug_mni.nii.gz
        """
        self.name = name
        self.glob_pattern = glob_pattern
        self.id_from_filename = id_from_filename

        self.collect_files()

    def collect_files(self):
        """
        Map file IDs to corresponding file paths, and file paths
        to file IDs.
        Raises an error for encountered invalid file names.
        """
        paths = glob.glob(self.glob_pattern)
        self.file_paths = []
        self.file_path_to_image_label = {}

        regexp = re.compile(self.id_from_filename["regexp"])
        group_id = self.id_from_filename["regex_id_group"]
        for p in paths:
            match = regexp.match(p)
            if match is None:
                warnings.warn("Could note extract id from path {}"
                              .format(p))
            else:
                file_id = match.group(group_id)
                self.file_paths.append(p)
                self.file_path_to_image_label[p] = file_id

    def get_file_paths(self):
        return self.file_paths

    def get_file_paths_to_id(self):
        return self.file_path_to_id

    def get_file_path(self, file_id):
        return self.id_to_file_path[file_id]

    def get_file_image_label(self, file_path):
        return self.file_path_to_image_label[file_path]
