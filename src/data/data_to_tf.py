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
import src.features as ft_def
from src.data.features_store import FeaturesStore
from src.data.data_aggregator import DataAggregator, UniqueLogger


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


class DataAggregatorToTFWriters(DataAggregator):
    def __init__(self, config, converted_dir, r):
        DataAggregator.__init__(
            self,
            config,
            r,
        )
        self.create_writers(converted_dir, **config['train_dataset_split'])

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

    def _add_image(self, image_path, features):
        Feature = tf.train.Feature
        Int64List = tf.train.Int64List

        # Transform features and write
        def _int64_to_feature(v):
            return Feature(int64_list=Int64List(value=[v]))

        def _str_to_feature(s):
            return Feature(bytes_list=tf.train.BytesList(
                value=[tf.compat.as_bytes(s)]
            ))

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

        img_data = nib.load(image_path).get_data()
        if list(img_data.shape) != list(self.config['image_shape']):
            self.add_error(
                image_path,
                'Image has shape %s, expected %s' % (
                    img_data.shape, self.config['image_shape'])
            )
            return False

        writer = self.get_writer_for_image(features)
        if writer is None:
            self.add_error(image_path, 'Image has no writer')
            return False

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
                return False
        return True

    def get_writer_for_image(self, features):
        train_or_test = self.get_sample_dataset(features)
        if train_or_test == 'test':
            return self.test_writer
        # Train set is sharded + splitted by feature
        shard_writers = self.r.choice(self.train_writers)
        for k, v in shard_writers.items():
            if k == '':
                continue
            assert(k in features)
            if features[k]:
                return v
        return shard_writers['']

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

    def finish(self):
        self.test_writer.close()
        for shard_writers in self.train_writers:
            for w in shard_writers.values():
                w.close()
        DataAggregator.finish(self)


class DataSource(object):
    def __init__(
        self,
        name,
        glob_pattern,
        patients_features,
        features_from_filename,
        seed=0,
    ):
        self.name = name
        self.all_files = glob.glob(glob_pattern)
        self.r = random.Random(seed)
        self.features_store = FeaturesStore(
            csv_file_path=patients_features,
            features_from_filename=features_from_filename,
        )

    def preprocess(self, dataset):
        dataset.begin_study(self.name, len(self.all_files))
        self.r.shuffle(self.all_files)

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
        DataSource(**source_config)
        for source_config in config['data_sources']
    ]


def process_all_files(config, dataset):
    data_sources = get_all_data_sources(config)
    for e in data_sources:
        e.preprocess(dataset)
    dataset.finish()


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
    process_all_files(config, DataAggregatorToTFWriters(
        config, converted_dir, random.Random(),
    ))
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
