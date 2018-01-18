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


def train_input(config_data_generation, config_data_streaming):
    dataset = tf.data.TFRecordDataset([
            config_data_generation['data_converted_directory'] +
            config_data_generation['train_database_file']
        ],
        compression_type=config_data_generation['dataset_compression'],
    )

    for func, args in config_data_streaming.items():
        if func == 'map':
            args['map_func'] = parser
        dataset = getattr(dataset, func)(**args)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def test_input(config_data_generation, config_data_streaming):
    dataset = tf.data.TFRecordDataset([
            config_data_generation['data_converted_directory'] +
            config_data_generation['train_database_file']
        ],
        compression_type=config_data_generation['dataset_compression'],
    )

    dataset = dataset.map(parser)
    dataset = dataset.batch(8)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()
