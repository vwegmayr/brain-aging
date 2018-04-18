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

    def finish_study(self):
        for img_data in self.current_study_images:
            if self._add_image(img_data[0], img_data[1]):
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
        self.current_study_images.append((image_path, features))

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
