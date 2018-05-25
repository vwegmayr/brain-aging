import abc
import csv
import glob
import re


class FileStream(abc.ABC):
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
        pass

    @abc.abstractmethod
    def group_data(self):
        pass

    @abc.abstractmethod
    def make_train_test_split(self):
        pass

    @abc.abstractmethod
    def load_sample(self):
        pass

    def parse_meta_csv(self):
        meta_info = {}
        with open(self.meta_csv) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                key = row[self.meta_id_column]
                meta_info[key] = row

        return meta_info

    def get_file_path(self, file_id):
        return self.file_id_to_meta[file_id]["file_path"]


class Group(object):
    def __init__(self, file_ids, is_train=None):
        self.file_ids = file_ids
        self.is_train = is_train

    def get_file_ids(self):
        return self.file_ids


class DataSource(object):
    def __init__(self, name, glob_pattern, id_from_filename):
        self.name = name
        self.glob_pattern = glob_pattern
        self.id_from_filename = id_from_filename

        self.collect_files()

    def collect_files(self):
        paths = glob.glob(self.glob_pattern)
        self.file_paths = []
        self.file_path_to_id = {}
        self.id_to_file_path = {}

        regexp = re.compile(self.id_from_filename["regexp"])
        group_id = self.id_from_filename["regex_id_group"]
        for p in paths:
            match = regexp.match(p)
            if match is None:
                raise ValueError("Could note extract id from path {}"
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
