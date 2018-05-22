import os
import numpy as np
import tensorflow as tf
import src.features as ft_def
from src.data.data_to_tf import generate_tf_dataset, iter_slices


# Functions for Data Provider interface
class DataProvider(object):
    def __init__(self, input_fn_config):
        input_fn_config['data_generation']['image_shape'] = \
            input_fn_config['image_shape']
        self.path = generate_tf_dataset(input_fn_config['data_generation'])
        self.input_fn_config = input_fn_config
        self.seed = 0

    def export_dataset(self):
        # Not implemented
        return None

    def get_input_fn(self, train, shard):
        self.seed += 1
        def _input_fn():
            return input_iterator(
                self.input_fn_config['data_generation'],
                self.input_fn_config['data_streaming'],
                data_path=self.path,
                shard=shard,
                type='train' if train else 'test',
                shuffle_seed=self.seed,
            )
        return _input_fn

    def predict_features(self, features):
        return random_crop(features)

    def get_mri_shape(self):
        for s in iter_slices(
            np.zeros(self.input_fn_config['image_shape']),
            self.input_fn_config['data_generation'],
        ):
            return list(s.shape)
        assert(False)
# End of interface functions


def parse_record(record):
    # MRI image shape should be set at this point (taken from generator config)
    assert(ft_def.all_features.feature_info[ft_def.MRI]['shape'] != [])
    keys_to_features = {
        name: tf.FixedLenFeature(shape=info['shape'], dtype=info['type'])
        for name, info in ft_def.all_features.feature_info.items()
    }

    parsed = tf.parse_single_example(record, features=keys_to_features)
    return parsed


def parser(record):
    parsed = parse_record(record)

    def process_feature(ft, ft_info):
        return tf.reshape(ft, ft_info['shape'])

    return {
        name: process_feature(parsed[name], info)
        for name, info in ft_def.all_features.feature_info.items()
    }


def random_crop(records):
    mri_shape = records[ft_def.MRI].get_shape().as_list()
    if len(mri_shape) == 4:
        # TODO: Predict mode, then batch_size = ?
        mri_shape[0] = 1
    mri_shape[-3] = 85
    mri_shape[-2] = 95
    mri_shape[-1] = 85
    records[ft_def.MRI] = tf.random_crop(
        records[ft_def.MRI],
        mri_shape,
    )
    return records


def gen_dataset_iterator(config, dataset):
    for func_call in config:
        func_call = func_call.copy()
        f = func_call['call']
        del func_call['call']
        dataset = getattr(dataset, f)(**func_call)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def first(a, b):
    return a


def second_second(a, b):
    return b[1]


def flat_map_concat_fn(class1, *args):
    d = tf.data.Dataset.from_tensors(class1)
    for c in args:
        d = d.concatenate(tf.data.Dataset.from_tensors(c))
    return d


def input_iterator(
    config_data_generation,
    config_data_streaming,
    data_path,
    shard=None,
    type='train',
    shuffle_seed=0,
):
    def get_dataset(filename):
        return tf.data.TFRecordDataset([os.path.join(
                data_path,
                filename,
            )],
            compression_type=config_data_generation['dataset_compression'],
        )
    assert(type in ['train', 'test'])
    if type == 'test':
        dataset = get_dataset(config_data_generation['test_database_file'])
    else:
        ENABLE_RESAMPLING_BATCH_BALANCE = False
        if ENABLE_RESAMPLING_BATCH_BALANCE:
            datasets = [
                get_dataset('train_healthy_s{shard}.tfrecord'.format(shard=shard[0])).shuffle(buffer_size=100, seed=shuffle_seed).map(lambda r: (tf.convert_to_tensor(0), r)),
                get_dataset('train_health_ad_s{shard}.tfrecord'.format(shard=shard[0])).shuffle(buffer_size=100, seed=shuffle_seed).map(lambda r: (tf.convert_to_tensor(1), r)),
                get_dataset('train_health_mci_s{shard}.tfrecord'.format(shard=shard[0])).shuffle(buffer_size=100, seed=shuffle_seed).map(lambda r: (tf.convert_to_tensor(2), r)),
            ]
            # Resample
            total = float(417 * 2 + 342 * 2 + 978)
            for i, d in enumerate(datasets):
                datasets[i] = d.apply(tf.contrib.data.rejection_resample(
                    first,
                    [0.33, 0.33, 0.33],
                    [2 * 417./total, 2 * 342./total, 978./total],
                    seed=0,
                )).map(second_second)
            # Concatenate
            dataset = tf.data.Dataset.zip(
                tuple(datasets)
            ).flat_map(flat_map_concat_fn)
        else:
            dataset = get_dataset('train_s{shard}.tfrecord'.format(
                shard=shard[0]
            )).shuffle(buffer_size=100, seed=shuffle_seed)
    dataset = dataset.map(parser, num_parallel_calls=8)
    return gen_dataset_iterator(config_data_streaming['dataset'], dataset)
