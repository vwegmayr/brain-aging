import tensorflow as tf
import features


def parse_record(record):
    # MRI image shape should be set at this point (taken from generator config)
    assert(features.all_features.feature_info[features.MRI]['shape'] != [])
    keys_to_features = {
        name: tf.FixedLenFeature(shape=info['shape'], dtype=info['type'])
        for (name, info) in features.all_features.feature_info.items()}

    parsed = tf.parse_single_example(record, features=keys_to_features)
    return parsed


def parser(record):
    parsed = parse_record(record)

    def process_feature(ft, ft_info):
        if ft_info['shape'] == []:
            return tf.reshape(ft, [1])
        return tf.reshape(ft, ft_info['shape'])

    return {
        name: process_feature(parsed[name], info)
        for (name, info) in features.all_features.feature_info.items()
    }


def distort(record):
    record[features.MRI] = tf.random_crop(record[features.MRI], [85, 95, 85])
    return record


def dataset_filter(record, keep_when_any_is_true=None):
    if keep_when_any_is_true is not None:
        cond = tf.constant(False)
        for field in keep_when_any_is_true:
            cond = tf.logical_or(
                cond,
                tf.equal(tf.reshape(record[field], []), 1),
            )
        return cond
    return True


def gen_dataset_iterator(config, dataset):
    for func_call in config:
        func_call = func_call.copy()
        f = func_call['call']
        del func_call['call']
        # Special cases
        if f == 'filter':
            # Forward all arguments to predicate
            kwargs = func_call.copy()
            func_call = {
                'predicate': lambda r: dataset_filter(r, **kwargs)
            }
        dataset = getattr(dataset, f)(**func_call)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def input_iterator(config_data_generation, config_data_streaming, shard=None, type='train'):
    assert(type in ['train', 'test'])
    dataset = tf.data.TFRecordDataset([
            config_data_generation['data_converted_directory'] +
            config_data_generation[type + '_database_file']
        ],
        compression_type=config_data_generation['dataset_compression'],
    )
    if shard is not None:
        dataset = dataset.shard(index=shard[0], num_shards=shard[1])
    return gen_dataset_iterator(config_data_streaming['dataset'], dataset)
