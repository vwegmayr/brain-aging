import random
import datetime
import abc
import src.features as ft_def
from modules.models.utils import custom_print


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
        self.count = 1
        self.train_test_split = {}
        self.total_files = total_files

    def get_sample_dataset(self, features):
        # Train/test dataset already defined
        if ft_def.DATASET in features:
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

        if self._add_image(image_path, features):
            self.stats[self.curr_study_name]['success'] += 1

    def add_error(self, path, message):
        self.stats[self.curr_study_name]['errors'].append(message)
        # print('%s [%s]' % (message, path))

    def finish(self):
        UniqueLogger.log('---- DATA CONVERSION STATS ----')
        for k, v in self.stats.items():
            UniqueLogger.log('%s: %d ok / %d errors' % (
                k, v['success'], len(v['errors'])))
            if len(v['errors']) > 0:
                UniqueLogger.log('    First error:')
                UniqueLogger.log('    %s' % v['errors'][0])
