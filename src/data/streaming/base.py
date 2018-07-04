import abc
import csv
import glob
import re
import warnings
import numpy as np
import tensorflow as tf
import copy
import os
from functools import reduce
import copy
from collections import OrderedDict

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
        self.config = config = copy.deepcopy(stream_config)
        self.meta_csv = config["meta_csv"]
        self.meta_id_column = config["meta_id_column"]
        self.batch_size = config["batch_size"]
        self.data_sources_list = config["data_sources"]
        self.seed = config["seed"]
        self.shuffle = config["shuffle"]
        self.silent = ("silent" in config and config["silent"])

        if "feature_collection" in config:
            self.feature_desc = _features.collections[
                config["feature_collection"]
            ].feature_info

        if self.seed is not None:
            self.np_random = np.random.RandomState(seed=self.seed)

        # Parse meta information
        self.file_id_to_meta = self.parse_meta_csv()
        csv_len = len(self.file_id_to_meta)

        self.name_to_data_source = OrderedDict()
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
            if not self.silent:
                print("{} has {} files".format(name, len(new_paths)))
            # Add path as meta information
            for p in new_paths:
                image_label = ds.get_file_image_label(p)
                if image_label in self.file_id_to_meta:
                    self.file_id_to_meta[p] = copy.deepcopy(
                        self.file_id_to_meta[image_label]
                    )
                    self.file_id_to_meta[p]["file_path"] = p
                    # store file name
                    # extract filename without extensions
                    file_name = os.path.split(p)[-1]
                    file_name = file_name.split(".")[0]
                    self.file_id_to_meta[p]["file_name"] = file_name
                    self.all_file_paths.append(p)
                else:
                    n_files_not_used += 1

        self.all_file_ids = set(self.all_file_paths)

        if not self.silent:
            print("{} files found but not specified meta csv"
                  .format(n_files_not_used))
            print("Number of files: {}".format(len(self.all_file_paths)))
            n_missing = csv_len - len(self.all_file_paths)
            print("Number of files missing: {}".format(n_missing))

        # Make train-test split
        train_ids, test_ids = self.make_train_test_split()
        all_train_test_ids = set(train_ids + test_ids)
        assert len(train_ids) + len(test_ids) == len(self.all_file_ids)
        # all filest are used
        assert len(self.all_file_ids.difference(all_train_test_ids)) == 0
        # Build train and test tuples
        self.groups = self.group_data(train_ids, test_ids)
        self.sample_shape = None

        self.sanity_checks()
        if not self.silent:
            print(">>>>>>>>> Sanity checks OK")

        # Print stats
        if not self.silent:
            train_groups = [group for group in self.groups if group.is_train]
            test_groups = [group for group in self.groups if not group.is_train]
            print(">>>>>>>> Train stats")
            self.print_stats(train_groups)
            print(">>>>>>>> Test stats")
            self.print_stats(test_groups)

    def get_batches(self, train=True):
        groups = [group for group in self.groups
                  if group.is_train == train]

        if self.shuffle:
            self.np_random.shuffle(groups)

        if self.batch_size == -1:
            return [[group for group in groups]]

        n_samples = len(groups)
        n_batches = int(n_samples / self.batch_size)

        batches = []
        for i in range(n_batches):
            bi = i * self.batch_size
            ei = (i + 1) * self.batch_size
            batches.append(groups[bi:ei])

        if n_batches * self.batch_size < n_samples:
            bi = n_batches * self.batch_size
            batches.append(groups[bi:])

        return batches

    @abc.abstractmethod
    def group_data(self, train_ids, test_ids):
        """
        Group files together that should be streamed together.
        For example groups of two files (i.e. pairs) or groups
        of three files (i.e. triples) can be formed.

        Args:
            - train_ids: list of file IDs that can
              be used for training
            - test_ids: list of file IDs that can be
              used for testing
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
    def load_sample(self, file_path):
        pass

    def dump_groups(self, outfolder, train, sep):
        groups = self.get_groups(train)

        if train:
            pref = "train"
        else:
            pref = "test"

        with open(os.path.join(outfolder, pref + "_groups.csv"), 'w') as f:
            for g in groups:
                f.write(sep.join(fid for fid in g.file_ids))
                f.write("\n")

    def dump_stats(self, outfolder):
        self.print_stats(
            self.get_groups(train=True),
            os.path.join(outfolder, "train_info.txt")
        )
        self.print_stats(
            self.get_groups(train=False),
            os.path.join(outfolder, "test_info.txt")
        )

    def dump_split(self, outfolder, sep="\t"):
        # one group per line
        # file IDs are tab separated
        self.dump_groups(outfolder, True, sep)
        self.dump_groups(outfolder, False, sep)
        self.dump_stats(outfolder)

    def dump_normalization(self, outfolder):
        pass

    def get_data_source_by_name(self, name):
        return self.name_to_data_source[name]

    def get_groups(self, train):
        return [g for g in self.groups if g.is_train == train]

    def get_set_file_ids(self, train=True):
        groups = [group for group in self.groups if group.is_train == train]
        fids = [fid for group in groups for fid in group.file_ids]
        return set(fids)

    def sanity_checks(self):
        """
        Check sound train-test split. The train set and test set
        of patients should be disjoint.
        """
        # Every group was assigned
        for group in self.groups:
            assert group.is_train in [True, False]

        # Disjoint patients
        train_ids = self.get_set_file_ids(True)
        test_ids = self.get_set_file_ids(False)

        if len(train_ids) == 0:
            ratio = 0
        else:
            ratio = len(train_ids)/(len(test_ids) + len(train_ids))

        if not self.silent:
            print("Achieved train ratio: {}".format(ratio))

        train_patients = set([self.get_patient_id(fid) for fid in train_ids])
        test_patients = set([self.get_patient_id(fid) for fid in test_ids])

        assert len(train_patients.intersection(test_patients)) == 0

    def print_stats(self, groups, outfile=None):
        group_size = len(self.groups[0].file_ids)

        dignosis_count = OrderedDict()
        ages = []
        age_diffs = []
        gender_0 = 0
        gender_1 = 0
        for group in groups:
            for fid in group.file_ids:
                ages.append(self.get_age(fid))
                diag = self.get_diagnose(fid)
                if diag not in dignosis_count:
                    dignosis_count[diag] = 0
                dignosis_count[diag] += 1

                g = self.get_gender(fid)
                if g == 0:
                    gender_0 += 1
                elif g == 1:
                    gender_1 += 1

            if group_size == 1:
                continue

            for i, fid in enumerate(group.file_ids[1:]):
                age1 = self.get_age(group.file_ids[i])
                age2 = self.get_age(fid)
                diff = abs(age1 - age2)
                age_diffs.append(diff)

        age_diffs = np.array(age_diffs)
        ages = np.array(ages)

        of = None
        if outfile is not None:
            of = open(outfile, 'w')
        else:
            print(">>>>>>>>>>>>>>>>")
        if len(ages) > 0:
            if of is None:
                print(">>>> Age stats, mean={}, std={}"
                      .format(np.mean(ages), np.std(ages)))
            else:
                of.write("Age stats, mean={}, std={}\n"
                         .format(np.mean(ages), np.std(ages)))

        if len(age_diffs) > 0:
            if of is None:
                print(">>>> Age diffences stats, mean={}, std={}"
                      .format(np.mean(age_diffs), np.std(age_diffs)))
            else:
                of.write("Age diffences stats, mean={}, std={}\n"
                         .format(np.mean(age_diffs), np.std(age_diffs)))

        total_diag_count = sum(dignosis_count.values())
        for diag, c in dignosis_count.items():
            if of is None:
                print(">>>> {} count: {} ({})"
                      .format(diag, c, c / total_diag_count))
            else:
                of.write("{} count: {} ({})\n"
                         .format(diag, c, c / total_diag_count))

        s = gender_0 * 1.0 + gender_1
        if s == 0:
            s = 0.00001
        if of is None:
            print(">>>> Gender 0: {} ({})".format(gender_0, gender_0 / s))
            print(">>>> Gender 1: {} ({})".format(gender_1, gender_1 / s))
        else:
            of.write("Gender 0: {} ({})\n".format(gender_0, gender_0 / s))
            of.write("Gender 1: {} ({})\n".format(gender_1, gender_1 / s))

        if of is not None:
            of.close()
        else:
            print(">>>>>>>>>>>>>>>>")

    def parse_meta_csv(self):
        """
        Parse the csv file containing the meta information about the
        data.
        """
        meta_info = OrderedDict()
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

    def get_test_retest_pairs(self, image_labels):
        # Groupy by patient
        image_labels = sorted(list(set(image_labels)))
        patient_to_image_labels = OrderedDict()
        for label in image_labels:
            patient = self.get_meta_info_by_key(label, "patient_label")
            if patient not in patient_to_image_labels:
                patient_to_image_labels[patient] = []
            patient_to_image_labels[patient].append(label)

        patients = sorted(list(patient_to_image_labels.keys()))
        groups = []
        for patient_label in patients:
            ids = patient_to_image_labels[patient_label]
            id_with_age = list(map(
                lambda x: (x, self.file_id_to_meta[x]["age"]),
                ids
            ))
            s = sorted(id_with_age, key=lambda x: (x[1], x[0]))

            # build pairs
            L = len(s)
            for i in range(1, L):
                id_1 = s[i - 1][0]
                id_2 = s[i][0]
                diag_1 = self.get_diagnose(id_1)
                diag_2 = self.get_diagnose(id_2)
                age_1 = self.get_age(id_1)
                age_2 = self.get_age(id_2)
                if (age_1 == age_2) and (diag_1 == diag_2):
                    assert id_1 != id_2
                    g = Group([id_1, id_2])
                    g.is_train = True
                    g.patient_label = patient_label
                    groups.append(g)

        return groups

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

    def get_patient_id(self, file_id):
        record = self.file_id_to_meta[file_id]
        return record["patient_label"]

    def get_age(self, file_id):
        record = self.file_id_to_meta[file_id]
        return record["age"]

    def get_gender(self, file_id):
        record = self.file_id_to_meta[file_id]
        return record["sex"]

    def get_patient_label(self, file_id):
        record = self.file_id_to_meta[file_id]
        return record["patient_label"]

    def get_sample_shape(self):
        assert len(self.groups) > 0
        some_id = self.groups[0].file_ids[0]
        path = self.get_file_path(some_id)
        sample = self.load_raw_sample(path)

        self.sample_shape = sample.shape
        return sample.shape

    def get_sample_1d_dim(self):
        shape = self.get_sample_shape()
        return reduce(lambda x, y: x * y, shape)

    def get_image_label(self, file_id):
        record = self.file_id_to_meta[file_id]
        return record["image_label"]

    def get_file_name(self, file_id):
        record = self.file_id_to_meta[file_id]
        return record["file_name"]

    def get_meta_info_by_key(self, file_id, key):
        record = self.file_id_to_meta[file_id]
        return record[key]

    def produce_groups(self, fids, group_size, train):
        if len(fids) == 0:
            return []

        groups = [[]]
        for fid in fids:
            last_group = groups[-1]
            if len(last_group) == group_size:
                groups.append([fid])
            else:
                last_group.append(fid)

        last_group = groups[-1]
        if len(last_group) < group_size:
            d = group_size - len(last_group)
            last_group += d * [last_group[-1]]

        groups = [Group(group_ids) for group_ids in groups]
        for group in groups:
            group.is_train = train

        return groups

    def make_one_sample_groups(self):
        groups = []
        for key in self.file_id_to_meta:
            if "file_path" in self.file_id_to_meta[key]:
                g = Group([key])
                g.patient_label = self.get_patient_label(key)
                groups.append(g)

        return groups

    def make_patient_groups(self):
        """
        One group per patient containing all file
        IDs.
        """
        patient_to_group = OrderedDict()
        for fid in self.file_id_to_meta:
            if "file_path" not in self.file_id_to_meta[fid]:
                continue
            patient = self.get_patient_id(fid)
            if patient not in patient_to_group:
                patient_to_group[patient] = Group([])
            patient_to_group[patient].add_file_id(fid)

        groups = []
        for k in patient_to_group.keys():
            groups.append(patient_to_group[k])

        return groups

    def get_patient_to_file_ids_mapping(self):
        patient_to_file_ids = OrderedDict()
        not_found = 0
        for file_id in self.file_id_to_meta:
            record = self.file_id_to_meta[file_id]
            if "file_path" in record:
                patient_label = record["patient_label"]
                if patient_label not in patient_to_file_ids:
                    patient_to_file_ids[patient_label] = []
                patient_to_file_ids[patient_label].append(
                    file_id
                )
            else:
                not_found += 1

        if not_found > 0:
            warnings.warn("{} files not found".format(not_found))

        return patient_to_file_ids

    def get_input_fn(self, train):
        batches = self.get_batches(train)
        groups = [group for batch in batches for group in batch]
        group_size = len(groups[0].file_ids)
        files = [group.file_ids for group in groups]

        # get feature names present in csv file (e.g. patient_label)
        # and added during preprocessing (e.g. file_name)
        feature_keys = self.file_id_to_meta[
            self.all_file_paths[0]
        ].keys()

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
                image = self.load_sample(path).astype(np.float32)
                ret += [image]

                ret += [
                    file_features[pf]
                    for pf in port_features
                ]
            # print("_read_files {}".format(ret[0] is None))
            return ret  # return list of features

        def _parser(*to_parse):
            if self.sample_shape is None:
                sample_shape = self.get_sample_shape()
            else:
                sample_shape = self.sample_shape
            el_n_features = 1 + len(port_features)  # sample + csv features
            all_features = OrderedDict()

            # parse features for every sample in group
            for i in range(group_size):
                self.feature_desc[_features.MRI]["shape"] = sample_shape
                mri_idx = i * el_n_features
                _mri = to_parse[mri_idx]
                ft = {
                    _features.MRI: tf.reshape(_mri, sample_shape),
                }

                ft.update({
                    port_features[i]: to_parse[mri_idx + i + 1]
                    for i in range(0, el_n_features - 1)
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
                el_features["X_" + str(i)] = el_features.pop(
                    _features.MRI + "_" + str(i)
                )
                all_features.update(el_features)

            return all_features

        labels = len(files) * [0]  # currently not used
        dataset = tf.data.Dataset.from_tensor_slices(
            tuple([files, labels])
        )

        # mri + other features
        read_types = group_size * ([tf.float32] + [
            self.feature_desc[fname]["type"]
            for fname in port_features
        ])

        num_calls = 4
        if "parallel_readers" in self.config:
            num_calls = self.config["parallel_readers"]
        dataset = dataset.map(
            lambda file_ids, label: tuple(tf.py_func(
                _read_files,
                [file_ids, label],
                read_types,
                stateful=False,
                name="read_files"
            )),
            num_parallel_calls=num_calls
        )

        prefetch = 4
        if "prefetch" in self.config:
            prefetch = self.config["prefetch"]
        dataset = dataset.map(_parser)
        dataset = dataset.prefetch(prefetch * self.config["batch_size"])
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
        self.label_stats = None

    def get_file_ids(self):
        return self.file_ids

    def add_file_id(self, fid):
        return self.file_ids.append(fid)

    def get_label_stats(self, streamer, categorical, numerical):
        if self.label_stats is not None:
            return self.label_stats

        stats = {
            "n": len(self.file_ids)
        }
        all_keys = categorical + numerical
        is_cat_bool = len(categorical) * [True] + len(numerical) * [False]

        for k, is_cat in zip(all_keys, is_cat_bool):
            stats[k] = {
                "count": 0,
                "vals": []
            }
            for fid in self.file_ids:
                val = streamer.get_meta_info_by_key(fid, k)
                stats[k]["vals"].append(val)
                if is_cat and val == 1:
                    stats[k]["count"] += 1
                if not is_cat:
                    stats[k]["count"] += val
            stats[k]["mean"] = stats[k]["count"] / stats["n"]
            if not is_cat:
                stats[k]["std"] = np.std(stats[k]["vals"])

        self.label_stats = stats
        return self.label_stats

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
        self.file_path_to_image_label = OrderedDict()

        regexp = re.compile(self.id_from_filename["regexp"])
        group_id = self.id_from_filename["regex_id_group"]
        discarded = 0
        for p in paths:
            match = regexp.match(p)
            if match is None:
                discarded += 1
                # warnings.warn("Could note extract id from path {}"
                #             .format(p))
            else:
                # extract image_label
                image_label = match.group(group_id)
                self.file_paths.append(p)
                self.file_path_to_image_label[p] = image_label

        if discarded > 0:
            warnings.warn("!!! {} IDs WERE NOT EXTRACTED !!!"
                          .format(discarded))

    def get_file_paths(self):
        return self.file_paths

    def get_file_paths_to_id(self):
        return self.file_path_to_id

    def get_file_path(self, file_id):
        return self.id_to_file_path[file_id]

    def get_file_image_label(self, file_path):
        return self.file_path_to_image_label[file_path]
