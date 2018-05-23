import csv
import re
import tensorflow as tf
import src.features as ft_def


def str_to_ft(ft, s):
    str_to_ft_funcs = {
        tf.int8: int,
        tf.int16: int,
        tf.int32: int,
        tf.int64: int,
        tf.uint8: int,
        tf.uint16: int,
        tf.uint32: int,
        tf.uint64: int,
        tf.string: str,
    }
    return str_to_ft_funcs[ft_def.all_features.feature_info[ft]['type']](s)


class FeaturesStore:
    """
    Stores features for a dataset
    """
    def __init__(
        self,
        csv_file_path,
        features_from_filename,
    ):
        self.load_patients_features(csv_file_path)
        self.features_from_filename = features_from_filename

        self.features_in_regexp = features_from_filename['features_group']
        assert(all([
            n in ft_def.all_features.feature_info
            for n in self.features_in_regexp
        ]))
        self.extract_from_path = re.compile(features_from_filename['regexp'])

    def load_patients_features(self, csv_file_path):
        """
        Load Patients features from csv
        This file should contain the following columns:
        - id
        - One column per feature from features.py file, excluding the ones
          extracted from the file name (typically study id and image id)
        """

        self.patients_ft = {}
        self.images_ft = {}
        self.image_label_ft = {}
        with open(csv_file_path) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Read features
                ft = {
                    col_name: str_to_ft(col_name, col_value)
                    for col_name, col_value in row.items()
                    if col_name in ft_def.all_features.feature_info
                }
                # Register them for later access
                if ft_def.STUDY_IMAGE_ID in ft:
                    image_id = ft[ft_def.STUDY_IMAGE_ID]
                    self.images_ft[image_id] = ft
                if ft_def.STUDY_PATIENT_ID in ft:
                    patient_id = ft[ft_def.STUDY_PATIENT_ID]
                    self.patients_ft[patient_id] = ft
                if ft_def.IMAGE_LABEL in ft:
                    image_label = ft[ft_def.IMAGE_LABEL]
                    self.image_label_ft[image_label] = ft

    def get_features_for_file(self, filename):
        # Add features from filename
        ft = {}
        match = self.extract_from_path.match(filename)
        if match is None:
            print(filename)
            raise LookupError('Regexp doesnt match')
        for ft_name, ft_group in self.features_in_regexp.items():
            ft[ft_name] = str_to_ft(ft_name, match.group(ft_group))
        # Add features from CSV - by image ID
        found_csv_entry = False
        if ft_def.STUDY_IMAGE_ID in ft:
            image_id = ft[ft_def.STUDY_IMAGE_ID]
            if image_id in self.images_ft:
                add_ft = self.images_ft[image_id].copy()
                add_ft.update(ft)
                ft = add_ft
                found_csv_entry = True
        # Or by image label
        if ft_def.IMAGE_LABEL in ft:
            image_label = ft[ft_def.IMAGE_LABEL]
            if image_label in self.image_label_ft:
                add_ft = self.image_label_ft[image_label].copy()
                add_ft.update(ft)
                ft = add_ft
                found_csv_entry = True
        # Or by patient ID
        if ft_def.STUDY_PATIENT_ID in ft:
            patient_id = ft[ft_def.STUDY_PATIENT_ID]
            if patient_id in self.patients_ft:
                add_ft = self.patients_ft[patient_id].copy()
                add_ft.update(ft)
                ft = add_ft
                found_csv_entry = True
        if not found_csv_entry:
            raise LookupError('No CSV features found')
        return ft
