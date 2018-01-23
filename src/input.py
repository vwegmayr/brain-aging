import tensorflow as tf
import features


def parser(record):
    # MRI image shape should be set at this point (taken from generator config)
    assert(features.all_features.feature_info[features.MRI]['shape'] != [])
    keys_to_features = {
        name: tf.FixedLenFeature(shape=info['shape'], dtype=info['type'])
        for (name, info) in features.all_features.feature_info.items()}

    parsed = tf.parse_single_example(record, features=keys_to_features)

    def process_feature(ft, ft_info):
        if ft_info['shape'] == []:
            return ft
        return tf.reshape(ft, ft_info['shape'])

    return {
        name: process_feature(parsed[name], info)
        for (name, info) in features.all_features.feature_info.items()
    }


def dataset_filter(record, keep_when_any_is_true=None):
    if keep_when_any_is_true is not None:
        cond = tf.constant(False)
        for field in keep_when_any_is_true:
            cond = cond or record[field] == 1
    return True


def gen_dataset_iterator(config, dataset):
    for func_call in config:
        func_call = func_call.copy()
        f = func_call['call']
        del func_call['call']
        if f == 'map':
            func_call['map_func'] = parser
        if f == 'filter':
            # Forward all arguments to predicate
            func_call = {
                'predicate': lambda r: dataset_filter(r, **func_call)
            }
        dataset = getattr(dataset, f)(**func_call)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def train_input(config_data_generation, config_data_streaming):
    dataset = tf.data.TFRecordDataset([
            config_data_generation['data_converted_directory'] +
            config_data_generation['train_database_file']
        ],
        compression_type=config_data_generation['dataset_compression'],
    )
    return gen_dataset_iterator(config_data_streaming['dataset'], dataset)


def test_input(config_data_generation, config_data_streaming):
    dataset = tf.data.TFRecordDataset([
            config_data_generation['data_converted_directory'] +
            config_data_generation['test_database_file']
        ],
        compression_type=config_data_generation['dataset_compression'],
    )
    disable_for_test = ['repeat']
    iter_config = [
        v
        for v in config_data_streaming['dataset']
        if v['call'] not in disable_for_test
    ]
    return gen_dataset_iterator(iter_config, dataset)
