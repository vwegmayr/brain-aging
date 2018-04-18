import random
import numpy as np
import datetime
import abc
import src.features as ft_def
from modules.models.utils import custom_print


class ImageNormalizationException(Exception):
    pass


class UniqueLogger:
    printed = set()

    @staticmethod
    def log(text):
        if text in UniqueLogger.printed:
            return
        UniqueLogger.printed.add(text)
        custom_print(text)


# Data filters
def merge_class_into(agg, from_feature, to_feature):
    update_count = 0
    for k in agg.file_to_features.keys():
        if agg.file_to_features[k][from_feature]:
            update_count += 1
            agg.file_to_features[k].update({
                from_feature: 0,
                to_feature: 1,
            })
    print('merge_class_into(%s -> %s): %d affected' % (
        from_feature, to_feature, update_count,
    ))


def map_patient_to_files(train_set, file_to_features):
    set_patient_to_images = {}
    for f in train_set:
        patient_id = file_to_features[f][ft_def.STUDY_PATIENT_ID]
        if patient_id not in set_patient_to_images:
            set_patient_to_images[patient_id] = []
        set_patient_to_images[patient_id].append(f)
    return set_patient_to_images

def modify_dataset(
    agg,
    comment,
    filter_only_class=None,
    ensure_all_patients_have_at_least_this_n_of_files=None,
    remove_data_augmentation=None,
    keep_patients=None,
    max_images_per_patient=None,
    min_images_per_patient=None,
    maximum_total_files=None,
    seed=0,
):
    file_to_features = agg.file_to_features
    dataset = agg.current_study_images
    if filter_only_class is not None:
        dataset = [f for f in dataset if file_to_features[f][filter_only_class]]
    agg.current_study_images = [
        f for f in agg.current_study_images
        if f not in dataset
    ]
    r = random.Random(seed)
    if remove_data_augmentation is not None:
        dataset = [
            f
            for f in dataset
            if remove_data_augmentation not in f
        ]
    # Group images by patient_id
    set_patient_to_images = map_patient_to_files(dataset, file_to_features)
    if ensure_all_patients_have_at_least_this_n_of_files is not None:
        set_patient_to_images = {
            p: l
            for p, l in set_patient_to_images.items()
            if len(l) >= ensure_all_patients_have_at_least_this_n_of_files
        }
    # For every patient, limit number of images
    if max_images_per_patient is not None:
        for patient_id in set_patient_to_images.keys():
            set_patient_to_images[patient_id].sort()
            r.shuffle(set_patient_to_images[patient_id])
            set_patient_to_images[patient_id] = \
                set_patient_to_images[patient_id][:max_images_per_patient]
    if min_images_per_patient is not None:
        set_patient_to_images = {
            k: v
            for k, v in set_patient_to_images.items()
            if len(v) >= min_images_per_patient
        }
    # Select patients
    take_patients = set_patient_to_images.keys()
    r.shuffle(take_patients)
    if keep_patients is not None:
        take_patients = take_patients[:keep_patients]
    total_files = 0
    for p in take_patients:
        total_files += len(set_patient_to_images[p])
    if maximum_total_files is None or maximum_total_files > total_files:
        maximum_total_files = total_files
    # Take all the files of selected patients
    max_reps = 0
    dataset = []
    while len(dataset) < maximum_total_files:
        # Add a file from every patient
        for patient_id in take_patients:
            if max_reps < len(set_patient_to_images[patient_id]):
                dataset.append(set_patient_to_images[patient_id][max_reps])
                if len(dataset) == maximum_total_files:
                    break
        max_reps += 1
    r.shuffle(dataset)
    # Debug print
    set_patient_to_images = map_patient_to_files(dataset, file_to_features)
    counts, number_patients = np.unique([
            len(v) for v in set_patient_to_images.values()
        ],
        return_counts=True,
    )
    print('  Dataset filtering: %s' % comment)
    for i in range(len(counts)):
        if i > 3:
            print('    ... up to %d samples each' % max_reps)
            break
        print('    %d patients with %d samples each' % (
            number_patients[i], counts[i],
        ))
    # Append some features
    agg.current_study_images += dataset

def set_dataset(agg, dataset):
    for f in agg.current_study_images:
        agg.file_to_features[f][ft_def.DATASET] = dataset


class DataAggregator:
    def __init__(self, config, r):
        self.config = config
        self.study_to_id = {}
        self.curr_study_id = -1
        self.curr_study_name = ''
        self.count = 0
        self.stats = {}
        self.r = r
        self.file_to_features = {}

    @abc.abstractmethod
    def _add_image(self, image_path, features):
        return

    def begin_study(self, study_name, total_files):
        self.study_to_id[study_name] = len(self.study_to_id)
        self.curr_study_id = self.study_to_id[study_name]
        self.curr_study_name = study_name
        self.stats[study_name] = {
            'success': 0,
            'errors': []
        }
        self.current_study_images = []
        self.count = 1
        self.train_test_split = {}
        self.total_files = total_files

    def finish_study(self, modifiers):
        ALL_MODIFIERS = {
            'merge_class_into': merge_class_into,
            'modify_dataset': modify_dataset,
            'set_dataset': set_dataset,
        }
        for m in modifiers:
            ALL_MODIFIERS[m['type']](self, **m['args'])

        for img_data in self.current_study_images:
            if self._add_image(img_data, self.file_to_features[img_data]):
                self.stats[self.curr_study_name]['success'] += 1

    def get_sample_dataset(self, features):
        # Train/test dataset already defined
        if ft_def.DATASET in features and features[ft_def.DATASET] != '':
            return features[ft_def.DATASET]

        ft_value = features[self.config['train_test_split_on_feature']]
        if ft_value not in self.train_test_split:
            if self.r.random() < self.config['test_set_size_ratio']:
                self.train_test_split[ft_value] = 'test'
            else:
                self.train_test_split[ft_value] = 'train'
        return self.train_test_split[ft_value]

    def pass_filters(self, features, any_is_true=None):
        if any_is_true is None:
            return True
        for ft in any_is_true:
            if features[ft]:
                return True
        return False

    def add_image(self, image_path, features):
        if self.count % (self.total_files / 10) == 1:
            UniqueLogger.log('%s: [%s] Processing image #%d/%d...' % (
                str(datetime.datetime.now()), self.curr_study_name,
                self.count, self.total_files))
        self.count += 1

        if 'filters' in self.config:
            if not self.pass_filters(features, **self.config['filters']):
                return

        features[ft_def.STUDY_ID] = self.curr_study_id

        # Check we have all features set
        for ft_name, ft_name_def in ft_def.all_features.feature_info.items():
            if (ft_name != ft_def.MRI and
                    ft_name not in features):
                if ft_name_def['default'] is None:
                    UniqueLogger.log('[FATAL] Feature `%s` missing for %s' % (
                        ft_name, image_path))
                    assert(False)
                features[ft_name] = ft_name_def['default']

        self.file_to_features[image_path] = features
        self.current_study_images.append(image_path)

    def add_error(self, path, message):
        self.stats[self.curr_study_name]['errors'].append(message)
        # print('%s [%s]' % (message, path))

    def finish(self):
        UniqueLogger.log('---- DATA CONVERSION STATS ----')
        for k, v in self.stats.items():
            errors, errors_count = np.unique(v['errors'], return_counts=True)
            UniqueLogger.log('%s: %d ok / %d errors' % (
                k, v['success'], len(v['errors'])))
            for e, e_c in zip(errors, errors_count):
                UniqueLogger.log('  %4d times:   %s' % (e_c, e))

    # Image data processing
    def _norm(self, mri, enable, outlier_percentile):
        if not enable:
            return mri
        max_value = np.percentile(mri, outlier_percentile)
        mri_for_stats = np.copy(mri)
        mri_for_stats[mri > max_value] = max_value
        m = np.mean(mri_for_stats)
        std = np.std(mri_for_stats)
        if std < 0.01:
            raise ImageNormalizationException()
        return (mri - m) / std

    def process_image_data(self, image_path, image_data):
        image_data = self._norm(
            image_data,
            **self.config['image_normalization']
        )
        return image_data
