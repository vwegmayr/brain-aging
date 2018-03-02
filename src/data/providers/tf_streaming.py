import os
import tensorflow as tf
import src.features as ft_def
from src.data.data_to_tf import generate_tf_dataset


# Functions for Data Provider interface
class DataProvider(object):
    def __init__(self, input_fn_config):
        input_fn_config['data_generation']['image_shape'] = \
            input_fn_config['image_shape']
        self.path = generate_tf_dataset(input_fn_config['data_generation'])
        self.input_fn_config = input_fn_config

    def get_input_fn(self, train, shard):
        def _input_fn():
            return input_iterator(
                self.input_fn_config['data_generation'],
                self.input_fn_config['data_streaming'],
                data_path=self.path,
                shard=shard,
                type='train' if train else 'test',
            )
        return _input_fn

    def predict_features(self, features):
        return random_crop(features)
# End of interface functions


def parse_record(record):
    # MRI image shape should be set at this point (taken from generator config)
    assert(ft_def.all_features.feature_info[ft_def.MRI]['shape'] != [])
    keys_to_features = {
        name: tf.FixedLenFeature(shape=info['shape'], dtype=info['type'])
        for (name, info) in ft_def.all_features.feature_info.items()}

    parsed = tf.parse_single_example(record, features=keys_to_features)
    return parsed


def parser(record):
    parsed = parse_record(record)

    def process_feature(ft, ft_info):
        return tf.reshape(ft, ft_info['shape'])

    return {
        name: process_feature(parsed[name], info)
        for (name, info) in ft_def.all_features.feature_info.items()
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


def input_iterator(
    config_data_generation,
    config_data_streaming,
    data_path,
    shard=None,
    type='train',
):
    assert(type in ['train', 'test'])
    dataset = tf.data.TFRecordDataset([os.path.join(
            data_path,
            config_data_generation[type + '_database_file'],
        )],
        compression_type=config_data_generation['dataset_compression'],
    )
    if shard is not None:
        dataset = dataset.shard(index=shard[0], num_shards=shard[1])
    return gen_dataset_iterator(config_data_streaming['dataset'], dataset)
