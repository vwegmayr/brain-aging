import abc
import csv
import glob
import re
import warnings


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

        # Parse meta information
        self.file_id_to_meta = self.parse_meta_csv()

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
                file_id = ds.get_file_id(p)
                if file_id in self.file_id_to_meta:
                    self.file_id_to_meta[file_id]["file_path"] = p
                    self.all_file_paths.append(p)
                else:
                    n_files_not_used += 1

        print("{} files found but not specified meta csv"
              .format(n_files_not_used))
        print("Number of files: {}".format(len(self.all_file_paths)))
        n_missing = len(self.file_id_to_meta) - len(self.all_file_paths)
        print("Number of files missing: {}".format(n_missing))

        # Group files into tuples
        self.groups = self.group_data()

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
                    if k == self.meta_id_column:
                        continue
                    if row[k].isdigit():
                        row[k] = int(row[k])

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
        self.file_path_to_id = {}
        self.id_to_file_path = {}

        regexp = re.compile(self.id_from_filename["regexp"])
        group_id = self.id_from_filename["regex_id_group"]
        for p in paths:
            match = regexp.match(p)
            if match is None:
                raise warnings.warn("Could note extract id from path {}"
                                    .format(p))
            else:
                file_id = match.group(group_id)
                self.file_paths.append(p)
                self.id_to_file_path[file_id] = p
                self.file_path_to_id[p] = file_id

    def get_file_paths(self):
        return self.file_paths

    def get_file_paths_to_id(self):
        return self.file_path_to_id

    def get_file_path(self, file_id):
        return self.id_to_file_path[file_id]

    def get_file_id(self, file_path):
        return self.file_path_to_id[file_path]
