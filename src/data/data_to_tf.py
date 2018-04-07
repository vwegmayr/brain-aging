"""
!!! WARNING !!!
When using in this script external values (config keys, files, ...)
make sure their content appears somehow in @get_data_preprocessing_values
function. So that when their value change, data is regenerated automatically.
"""

import numpy as np
import tensorflow as tf
import random
import json
import sys
import hashlib
import os
import glob
import re
import datetime
import nibabel as nib
from modules.models.utils import custom_print
import src.features as ft_def
from src.data.features_store import FeaturesStore


class UniqueLogger:
    printed = set()

    @staticmethod
    def log(text):
        if text in UniqueLogger.printed:
            return
        UniqueLogger.printed.add(text)
        custom_print(text)


class ImageNormalizationException(Exception):
    pass


def get_data_preprocessing_values(config):
    """
    Returns a dictionnary with serializable values.
    Whenever the data should be re-generated, the content
    of the returned dictionnary should change.
    """
    class MyEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, re._pattern_type):
                return {}
            return o.__dict__
    return {
        'sources': [
            MyEncoder().encode(s.__dict__)
            for s in get_all_data_sources(config)
        ],
        'config': config,
        'modules': {
            'tf_major_version': tf.__version__.split('.')[0],
            'extractor_version': 1,
        }
    }


def get_3d_array_dim(a, dim, dim_val):
    assert(dim in [0, 1, 2])
    if dim == 0:
        return a[dim_val, :, :]
    elif dim == 1:
        return a[:, dim_val, :]
    return a[:, :, dim_val]


def iter_slices(img_data, config):
    if 'image_slices' in config:
        for s in config['image_slices']:
            yield get_3d_array_dim(img_data, s['dimension'], s['value'])
    elif 'image_crop' in config:
        image_crop = config['image_crop']
        assert(len(image_crop) == 6)
        yield img_data[
            image_crop[0]:image_crop[1],
            image_crop[2]:image_crop[3],
            image_crop[4]:image_crop[5],
        ]
    else:
        yield img_data


class DataAggregator:
    def __init__(self, config, converted_dir):
        self.config = config
        self.study_to_id = {}

        self.create_writers(converted_dir, **config['train_dataset_split'])
        self.curr_study_id = -1
        self.curr_study_name = ''
        self.count = 0
        self.stats = {}

    def create_writers(self, converted_dir, split_features=[], num_shards=1):
        compression = getattr(
            tf.python_io.TFRecordCompressionType,
            self.config['dataset_compression'],
        )

        def create_writer(filename):
            return tf.python_io.TFRecordWriter(
                os.path.join(converted_dir, filename),
                tf.python_io.TFRecordOptions(compression),
            )
        self.test_writer = create_writer(self.config['test_database_file'])
        self.train_writers = []
        for shard in range(num_shards):
            shard_writers = {
                '': create_writer(self.config['train_database_file'].format(
                            feature='', shard=shard,
                    ))
            }
            shard_writers.update({
                ft: create_writer(self.config['train_database_file'].format(
                            feature=ft, shard=shard,
                    ))
                for ft in split_features
            })
            self.train_writers.append(shard_writers)

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
            if random.random() < self.config['test_set_size_ratio']:
                self.train_test_split[ft_value] = 'test'
            else:
                self.train_test_split[ft_value] = 'train'
        return self.train_test_split[ft_value]

    def get_writer_for_image(self, features):
        train_or_test = self.get_sample_dataset(features)
        if train_or_test == 'test':
            return self.test_writer
        # Train set is sharded + splitted by feature
        shard_writers = random.choice(self.train_writers)
        for k, v in shard_writers.items():
            if k == '':
                continue
            assert(k in features)
            if features[k]:
                return v
        return shard_writers['']

    def pass_filters(self, features, any_is_true=None):
        if any_is_true is None:
            return True
        for ft in any_is_true:
            if features[ft]:
                return True
        return False

    def add_image(self, image_path, features):
        Feature = tf.train.Feature
        Int64List = tf.train.Int64List

        if self.count % (self.total_files / 10) == 1:
            UniqueLogger.log('%s: [%s] Processing image #%d/%d...' % (
                str(datetime.datetime.now()), self.curr_study_name,
                self.count, self.total_files))
        self.count += 1

        if 'filters' in self.config:
            if not self.pass_filters(features, **self.config['filters']):
                return

        img_data = nib.load(image_path).get_data()
        if list(img_data.shape) != list(self.config['image_shape']):
            self.add_error(
                image_path,
                'Image has shape %s, expected %s' % (
                    img_data.shape, self.config['image_shape'])
            )
            return

        writer = self.get_writer_for_image(features)
        if writer is None:
            self.add_error(image_path, 'Image has no writer')
            return

        # Transform features and write
        def _int64_to_feature(v):
            return Feature(int64_list=Int64List(value=[v]))

        def _str_to_feature(s):
            return Feature(bytes_list=tf.train.BytesList(
                value=[tf.compat.as_bytes(s)]
            ))
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

        img_features = {
            k: _int64_to_feature(v)
            for k, v in features.items()
            if ft_def.all_features.feature_info[k]['type'] != tf.string
        }
        img_features.update({
            k: _str_to_feature(v)
            for k, v in features.items()
            if ft_def.all_features.feature_info[k]['type'] == tf.string
        })

        for s in iter_slices(img_data, self.config):
            try:
                self._write_image(
                    writer,
                    s,
                    img_features,
                    image_path,
                )
            except ImageNormalizationException:
                self.add_error(
                    image_path,
                    'Unable to normalize image',
                )
                return

        self.stats[self.curr_study_name]['success'] += 1

    def _write_image(self, writer, img_data, img_features, image_path):
        img_data = self._norm(
            img_data,
            **self.config['image_normalization']
        )
        img_features[ft_def.MRI] = tf.train.Feature(
            float_list=tf.train.FloatList(
                value=img_data.reshape([-1])
            ),
        )
        assert(all([
            ft_name in img_features
            for ft_name, ft_info in ft_def.all_features.feature_info.items()
        ]))

        example = tf.train.Example(
            features=tf.train.Features(feature=img_features),
        )
        writer.write(example.SerializeToString())

    def add_error(self, path, message):
        self.stats[self.curr_study_name]['errors'].append(message)
        print('%s [%s]' % (message, path))

    def finish(self):
        self.test_writer.close()
        for shard_writers in self.train_writers:
            for w in shard_writers.values():
                w.close()

        UniqueLogger.log('---- DATA CONVERSION STATS ----')
        for k, v in self.stats.items():
            UniqueLogger.log('%s: %d ok / %d errors' % (
                k, v['success'], len(v['errors'])))
            if len(v['errors']) > 0:
                UniqueLogger.log('    First error:')
                UniqueLogger.log('    %s' % v['errors'][0])

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


class DataSource(object):
    def __init__(self, config):
        self.config = config
        self.all_files = glob.glob(config['glob'])
        self.features_store = FeaturesStore(
            csv_file_path=config['patients_features'],
            features_from_filename=config['features_from_filename'],
        )

    def preprocess(self, dataset):
        dataset.begin_study(self.config['name'], len(self.all_files))
        random.shuffle(self.all_files)

        # MRI scans
        for f in self.all_files:
            try:
                ft = self.features_store.get_features_for_file(f)
                dataset.add_image(f, ft)
            except LookupError as e:
                dataset.add_error(f, str(e))
            except IOError as e:
                dataset.add_error(f, str(e))


def get_all_data_sources(config):
    return [
        DataSource(source_config)
        for source_config in config['data_sources']
    ]


def preprocess_all(config, converted_dir):
    random_state = random.getstate()
    random.seed(config['test_set_random_seed'])
    dataset = DataAggregator(config, converted_dir)
    data_sources = get_all_data_sources(config)
    for e in data_sources:
        e.preprocess(dataset)
    dataset.finish()
    random.setstate(random_state)


def generate_tf_dataset(config):
    """
    Saves data to
    $data_converted_directory/{hash}/...
    And return this directory
    """
    def obj_hash(obj):
        return hashlib.sha1(json.dumps(
            obj,
            ensure_ascii=False,
            sort_keys=True,
        )).hexdigest()[:8]
    current_extractor_values = get_data_preprocessing_values(config)
    h = obj_hash(current_extractor_values)
    converted_dir = os.path.join(config['data_converted_directory'], h)
    extraction_finished_file = os.path.join(converted_dir, "done.json")
    if os.path.isfile(extraction_finished_file):
        UniqueLogger.log(
            '[INFO] Extracted TF Dataset `' + converted_dir +
            '` is up-to-date. Skipping dataset generation :)'
        )
        return converted_dir
    UniqueLogger.log(
        '[INFO] TF Dataset in `' + converted_dir + '` is inexistant. ' +
        'Will generate from scratch.'
    )
    try:
        os.mkdir(converted_dir)
    except OSError:
        UniqueLogger.log(
            '[FATAL] Directory already exists. Maybe another process ' +
            'is currently generating data? If not, rm folder and try again.'
        )
        sys.exit(42)
    json.dump(
        current_extractor_values,
        open(os.path.join(converted_dir, "extractor_values.json"), "wb"),
        ensure_ascii=False,
        sort_keys=True,
        indent=4,
    )

    extract_start = datetime.datetime.now()
    preprocess_all(config, converted_dir)
    extract_end = datetime.datetime.now()
    json.dump({
            'start_time': str(extract_start),
            'end_time': str(extract_end),
            'elapsed': str(extract_end - extract_start),
        },
        open(extraction_finished_file, "wb"),
        ensure_ascii=False,
        sort_keys=True,
        indent=4,
    )
    return converted_dir
